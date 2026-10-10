# FastWAM 可视化工具

`Visualize/` 中的工具分为两类：

- **在线预测可视化**：运行一个 LIBERO episode，保存 DreamFastWAM / RoutedWAM 在推理时预测的未来模态，再统一渲染。
- **离线数据可视化**：读取 LeRobot 数据集已有的 RGB 和 extras，查看监督目标或生成点云。

所有命令均应在仓库根目录运行：

```bash
cd /mnt/hwdata/txc/FastWAM
conda activate fastwam
```

## 文件概览

| 文件 | 类型 | 用途 |
| --- | --- | --- |
| `infer_dream_episode.py` | 命令行入口 | 用 checkpoint 完整运行一个 LIBERO episode，保存并渲染模型预测的未来模态 |
| `dream_prediction_visualization.py` | Python 接口 | 读取 `infer_dream_episode.py` 保存的原始预测并生成 PNG/MP4 |
| `visualize_frame_extras.py` | 命令行入口 | 可视化数据集中的 RGB、dyn、depth、DINO、SAM 监督数据 |
| `depth_anything_to_pointcloud.py` | 命令行入口 | 将 Depth Anything metric depth 反投影为 PLY 点云 |
| `vggt_rgb_to_geometry.py` | 命令行入口 | 独立运行 VGGT 双视角 RGB 基线，保存 depth、相机参数和点云 |
| `modality_visualization.py` | Python 接口 | 通用的 depth、dynamic、DINO、SAM 可视化函数 |

## 1. 可视化模型预测的未来模态

入口为 `Visualize/infer_dream_episode.py`，配置文件为
`configs/visualize_dream_episode.yaml`。

该脚本只运行指定任务的一个初始状态，不会遍历整个测试集。每次重新规划时：

1. 在 Action 去噪前计算 Video / Dream，缓存逐层 K/V；
2. DreamFastWAM 做一次回归前向；生成式 RoutedWAM 执行配置指定步数的 Dream 去噪，随后 Action 多步去噪复用 K/V；
3. 将原始预测先保存到 CPU；
4. episode 完成后统一拟合颜色映射并生成可视化。

这样不会在每个 Action denoising step 重复运行 Dream decoder。生成式
RoutedWAM 的每个 Dream 去噪步仍需运行 decoder 预测速度，保存和渲染的是
最后一次 scheduler 更新后的目标空间预测，而不是速度、K/V 或真实未来标签。
PCA、KMeans 和视频编码统一在 episode 完成后执行。

### RoutedWAM 推理可视化

同一个入口支持 RoutedWAM，无需单独脚本。默认从 checkpoint 附近的训练
`config.yaml` 恢复模型类型和 Dream 步数，第一阶段、蒸馏阶段权重均可使用：

```bash
python Visualize/infer_dream_episode.py \
  ckpt=/path/to/routed_wam/checkpoints/weights/step_XXXXXX.pt \
  EVALUATION.task_suite_name=libero_goal \
  EVALUATION.task_id=0 \
  EVALUATION.initial_state_index=0 \
  EVALUATION.dream_inference_steps=1 \
  EVALUATION.num_inference_steps=10 \
  EVALUATION.output_dir=./evaluate_results/routed_dream/steps1
```

`EVALUATION.dream_inference_steps=8` 可改为 8 步 Dream 去噪；
`EVALUATION.num_inference_steps` 独立控制 Action。省略 Dream 参数或设为
`null` 会保留训练配置中的值。该覆盖在恢复训练配置之后应用，避免被覆盖掉；
不要用 `model.dream_scheduler.inference_steps` 代替它。
普通回归 Dream 不接受 Dream 去噪步数覆盖。

输出格式与原脚本一致：`raw_predictions/replan_*.pt` 保存所有 horizon 的
depth、dyn、DINO、SAM（以模型实际启用的模态为准），`rendered/` 保存双视角
PNG 和 MP4。record 的 `metadata` 和 `episode_manifest.json` 额外记录模型类、
Dream 预测方式、Dream 和 Action 的实际推理步数。回归 Dream 的去噪步数为 `null`。
这里展示的是最终多模态预测，不包含每个 Dream 去噪中间步或路由热力图。

检查预测本身的空间结构时，可加 `VISUALIZATION.overlay_alpha=1.0` 去掉
RGB 叠加，并用 `VISUALIZATION.sam_render_mode=pca` 查看 SAM 特征 PCA。
默认 `regions` 只是 SAM embedding 的 KMeans 聚类，不是 SAM mask decoder
输出的分割结果；即使特征是噪声，聚类也会分配彩色区域。
回归 Dream 的 dyn 经 sigmoid 显示；生成式 Dream 的最终 dyn 样本直接按
`[0,1]` 截断显示，不重复 sigmoid。原始 `.pt` 数值保持不变。
depth 中非正值显示为黑色。可视化不能保证生成预测具有正确的几何或语义。

### 基本命令

```bash
CUDA_VISIBLE_DEVICES=4 \
python Visualize/infer_dream_episode.py \
  ckpt=runs/dream_fastwam_libero/RUN/checkpoints/weights/step_008680.pt \
  EVALUATION.task_suite_name=libero_goal \
  EVALUATION.task_id=0 \
  EVALUATION.initial_state_index=0 \
  EVALUATION.output_dir=./evaluate_results/dream_episode/example
```

常用覆盖参数：

```bash
# 最多执行 100 个环境 step，每 10 个动作重新规划
EVALUATION.max_steps=100
EVALUATION.replan_steps=10

# Action flow-matching 的推理步数
EVALUATION.num_inference_steps=10

# 只保存原始预测，暂不渲染
VISUALIZATION.render_after_inference=false

# 调整输出尺寸和可视化开销
VISUALIZATION.panel_size=160
VISUALIZATION.sam_clusters=8
VISUALIZATION.max_projection_samples=2048
```

完整默认参数见 [`configs/visualize_dream_episode.yaml`](../configs/visualize_dream_episode.yaml)。

### checkpoint 配置和统计量

默认 `EVALUATION.use_training_config=true`。脚本会沿 checkpoint 所在的 run
目录自动寻找训练时保存的 `config.yaml`，用它恢复模型和数据配置，并寻找
`dataset_stats.json`。因此，正常情况下不需要手动重复训练配置。

如果 checkpoint 被单独移动，或自动发现失败，可显式指定：

```bash
python Visualize/infer_dream_episode.py \
  ckpt=/path/to/step_008680.pt \
  EVALUATION.training_config_path=/path/to/config.yaml \
  EVALUATION.dataset_stats_path=/path/to/dataset_stats.json \
  EVALUATION.task_suite_name=libero_goal \
  EVALUATION.task_id=0
```

训练配置必须与 checkpoint 的模型结构一致，尤其是 Dream modalities、
future offsets、query 数量和 decoder 设置。

### 输出目录

```text
example/
├── resolved_config.yaml
├── training_config_path.txt
├── episode_manifest.json
├── raw_predictions/
│   ├── replan_0000.pt
│   └── ...
├── rollout/
│   └── *.mp4
└── rendered/
    ├── dream_predictions.mp4
    ├── render_manifest.json
    └── frames/
        ├── replan_0000.png
        └── ...
```

- `raw_predictions/replan_XXXX.pt` 保存当前两路 RGB、proprio、预测 action、
  Dream predictions、future offsets 和相机 token 划分。
- `episode_manifest.json` 保存任务、checkpoint、成功状态、模态和运行时间等信息。
- `rollout/*.mp4` 是环境 rollout。
- `rendered/dream_predictions.mp4` 是未来模态预测；其中每一帧对应一次
  **replan**，并非一个 simulator step。

渲染器会按 checkpoint 实际包含的模态绘制。当前双视角预测的典型单个
horizon 形状为：

| 模态 | 合并后的预测形状 | 每个视角 |
| --- | --- | --- |
| depth | `[128, 256]` | `128×128` 完整 depth map |
| dyn | `[392, 1]` | `14×14` 动态网格 |
| DINO | `[16, 32, 768]` | `16×16×768` |
| SAM | `[16, 32, 256]` | `16×16×256` |

其中 depth、DINO 和 SAM 沿宽度方向拼接；dyn 以两个视角的扁平位置保存。
第一半是主视角 `image`，第二半是腕部视角 `wrist_image`。

## 2. 重新渲染已保存的 Dream 预测

`dream_prediction_visualization.py` 不是独立命令行脚本，而是
`infer_dream_episode.py` 使用的离线渲染接口。已经保存
`raw_predictions/` 后，可以在 Python 中重新渲染，不必再次运行环境或模型：

```bash
python - <<'PY'
from pathlib import Path
from Visualize.dream_prediction_visualization import render_saved_episode

result = render_saved_episode(
    Path("evaluate_results/dream_episode/example/raw_predictions"),
    Path("evaluate_results/dream_episode/example/rendered_retry"),
    fps=5,
    panel_size=224,
    alpha=0.5,
    sam_clusters=8,
    draw_contours=True,
    max_projection_samples=2048,
)
print(result)
PY
```

主要接口：

- `split_prediction_views()`：把合并预测拆成主视角和腕部视角。
- `load_episode_records()`：加载一个 episode 的所有 `replan_*.pt`。
- `fit_episode_projections()`：在整个 episode、所有 horizon 和两个视角上
  统一拟合 depth 范围、DINO PCA 和 SAM KMeans，保证各帧颜色稳定。
- `render_record_grid()`：渲染单个 replan。
- `render_saved_episode()`：批量输出 PNG 和 MP4。

## 3. 可视化数据集中的监督信息

`visualize_frame_extras.py` 读取数据集中的真实 RGB 和预先生成的 extras。
它展示的是训练 target/离线特征，**不是模型预测结果**。

默认数据路径为：

```text
data/libero_mujoco3.3.2/{suite}_no_noops_lerobot
```

支持 `libero_10`、`libero_goal`、`libero_spatial` 和 `libero_object`。

### 单帧

```bash
python Visualize/visualize_frame_extras.py \
  --dataset libero_goal \
  --episode 0 \
  --frame 50 \
  --output Visualize/results/frame_ep0_f50.png
```

### 整个 episode

只要输出后缀为 `.mp4`，或显式传入 `--all-frames`，脚本就会进入视频模式：

```bash
python Visualize/visualize_frame_extras.py \
  --dataset libero_goal \
  --episode 0 \
  --output Visualize/results/episode_0.mp4 \
  --fps 10
```

快速检查前 50 帧：

```bash
python Visualize/visualize_frame_extras.py \
  --dataset libero_goal \
  --episode 0 \
  --output Visualize/results/episode_0_preview.mp4 \
  --max-frames 50 \
  --panel-size 160 \
  --no-contours
```

主要参数：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--episode` | episode 索引 | `0` |
| `--frame` | 单帧模式下的帧索引 | `0` |
| `--cameras` | 要显示的相机 | `image wrist_image` |
| `--panel-size` | 每个面板的边长 | `300` |
| `--fps` | 输出视频帧率 | `10` |
| `--max-frames` | 视频模式最多渲染多少帧 | 不限制 |
| `--overlay-alpha` | overlay 透明度 | `0.48` |
| `--sam-clusters` | SAM embedding 的 KMeans 区域数 | `8` |
| `--no-contours` | 不绘制 SAM 区域边界 | 关闭 |
| `--no-extras` | 只显示 RGB | 关闭 |

extras 目录不存在时，脚本会给出警告并退化为仅显示 RGB。

## 4. Depth Anything 深度转点云

`depth_anything_to_pointcloud.py` 将一个数据帧的 metric depth 反投影为
ASCII PLY，不依赖 Open3D。

### 单相机点云

```bash
python Visualize/depth_anything_to_pointcloud.py \
  --dataset-root data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot \
  --episode 0 \
  --frame 50 \
  --cameras image \
  --stride 4 \
  --output Visualize/results/goal_ep0_f50_image.ply
```

默认使用垂直视场角 `--fovy 45` 估计相机内参。若有标定值，建议改为：

```bash
--intrinsics FX FY CX CY
```

如不需要 RGB 颜色，可以加入 `--no-color`；否则脚本通过 `ffmpeg` 从对应
视频中解码 RGB 帧。还可以用 `--min-depth` 和 `--max-depth` 过滤深度范围。

### 双相机融合

融合主视角和腕部视角时，必须为每个相机提供 camera-to-world 位姿：

```bash
python Visualize/depth_anything_to_pointcloud.py \
  --dataset-root data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot \
  --episode 0 \
  --frame 50 \
  --cameras image wrist_image \
  --pose image=/path/to/agent_camera_to_world.npy \
  --pose wrist_image=/path/to/wrist_camera_to_world.npy \
  --output Visualize/results/goal_ep0_f50_fused.ply
```

位姿文件支持 NPY、NPZ、JSON 和文本格式，内容可以是固定的 `[4,4]`，
也可以是逐帧的 `[T,4,4]`。双相机融合缺少任意一路位姿时脚本会拒绝运行，
避免把两个不同坐标系中的点直接拼接。

单相机且不提供位姿时，输出坐标系为相机坐标系：`x` 向右、`y` 向下、
`z` 向前。

### DreamFastWAM 预测 depth

脚本也可通过 `--record` 直接读取 `replan_XXXX.pt` 中的
`dream_predictions["depth"]`。使用 `--future-offset` 选择预测时刻：

```bash
python Visualize/depth_anything_to_pointcloud.py \
  --record evaluate_results/dream_episode/example/raw_predictions/replan_0000.pt \
  --future-offset 16 \
  --cameras image \
  --fovy 45 \
  --no-color \
  --output Visualize/results/dream_depth_t16_image.ply
```

`--future-offset` 必须是 record 的 `future_offsets` 中存在的值。也可通过
`--horizon-index 0` 按下标选择，省略这两个参数时默认使用第 0 个 horizon。
Dream record 没有保存未来 RGB，因此推荐使用 `--no-color`；不加该参数时，
点的颜色来自当前观测 RGB，并不与未来 depth 严格对应。

主视角和腕部视角的 depth 属于各自相机坐标系。若要通过
`--cameras image wrist_image` 融合，仍必须用两个 `--pose` 参数提供所选未来
时刻的 camera-to-world 位姿。

## 5. 独立 VGGT 双视角几何基线

`vggt_rgb_to_geometry.py` 不读取 DreamFastWAM checkpoint，也不调用 Dream
Expert。它既可接收普通 RGB 图片路径，也可直接读取 Dream episode 保存的
`replan_XXXX.pt` 中的 `record["rgb"]`。脚本按照官方 VGGT 接口同时输入主视角
和腕部视角，预测 depth、depth confidence、point-map、point confidence，
以及相机内外参。

先安装[官方 VGGT](https://github.com/facebookresearch/vggt)：

```bash
git clone https://github.com/facebookresearch/vggt.git /path/to/vggt
pip install -e /path/to/vggt
```

然后运行：

```bash
CUDA_VISIBLE_DEVICES=4 \
python Visualize/vggt_rgb_to_geometry.py \
  --images /path/to/image.png /path/to/wrist_image.png \
  --view-names image wrist_image \
  --output-dir Visualize/results/vggt_pair
```

如果 RGB 来自 Dream episode 保存的某次 replan，可用 `--record` 直接读取
其中的 `image` 和 `wrist_image`，不需要导出中间 PNG：

```bash
CUDA_VISIBLE_DEVICES=4 \
python Visualize/vggt_rgb_to_geometry.py \
  --record evaluate_results/dream_episode/example/raw_predictions/replan_0000.pt \
  --view-names image wrist_image \
  --output-dir Visualize/results/vggt_pair
```

如果不希望安装 editable package，也可通过 `--vggt-root` 指向官方仓库：

```bash
CUDA_VISIBLE_DEVICES=4 \
python Visualize/vggt_rgb_to_geometry.py \
  --vggt-root /path/to/vggt \
  --images /path/to/image.png /path/to/wrist_image.png \
  --output-dir Visualize/results/vggt_pair
```

首次使用默认的 `facebook/VGGT-1B` 时会从 Hugging Face 下载权重。也可以
通过 `--model-id /path/to/local_model` 使用本地 `from_pretrained` 目录。

输出如下：

```text
vggt_pair/
├── predictions.npz
├── manifest.json
├── pointcloud_from_depth.ply
├── pointcloud_from_point_head.ply
└── depth/
    ├── image.png
    ├── image_input.png
    ├── wrist_image.png
    └── wrist_image_input.png
```

`predictions.npz` 保存未经着色的 VGGT 原始数值：

- `depth`、`depth_confidence`
- `point_map`、`point_confidence`
- `world_points_from_depth`
- `pose_encoding`、`extrinsic`、`intrinsic`
- VGGT resize/pad 后与预测逐像素对齐的 `processed_rgb`
- `view_names` 和原始图片路径

`pointcloud_from_depth.ply` 使用 depth、预测相机参数和 VGGT 官方
unprojection 生成；官方说明这种点云通常比 point-head 直接输出更加准确。
`pointcloud_from_point_head.ply` 同时保留，便于比较两个分支。默认每隔两个
像素保留一个点，并过滤 confidence 最低的 20%，可通过以下参数调整：

```bash
--point-stride 1
--confidence-percentile 0
```

VGGT 输出的坐标系和尺度不保证与 FastWAM depth target 相同。因此，数值
评测前应进行尺度/坐标对齐，不能只对两个 depth 数组直接相减。另外，
DreamFastWAM 预测的是 `t+offset` 的未来信息；若要做同一时刻的公平对比，
应向 VGGT 输入对应未来帧的两路真实 RGB，而不是当前观测。

## 6. 通用模态可视化接口

`modality_visualization.py` 不读取数据集，输入和输出均为 NumPy 数组，供
模型预测和离线 extras 共用。主要接口包括：

- `depth_colormap()`、`overlay_depth()`：metric depth 着色或叠加。
- `dynamic_map_from_tracks()`：将 CoTracker 位移转为运动强度。
- `dynamic_heatmap()`、`overlay_dynamic_heatmap()`：动态热力图。
- `overlay_sam_masks()`：绘制真实或预测 mask。
- `as_dense_features()`：将 token、CHW、HWC 或扁平特征转成 `[H,W,C]`。
- `fit_pca_projection()`、`feature_pca_rgb()`、`overlay_feature_pca()`：
  DINO/SAM dense feature 的 PCA 颜色映射。
- `fit_kmeans_projection()`、`kmeans_feature_regions()`、
  `overlay_embedding_regions()`：embedding 的 KMeans 区域可视化。

需要跨帧比较时，应先在整个序列上拟合一次 `PCAProjection` 或
`KMeansProjection`，然后把同一个 projection 传给每一帧；否则每帧独立
拟合会导致颜色跳变。

## 常见问题

### torchvision 的 video deprecation warning

类似下面的信息只是警告，不是运行失败：

```text
The video decoding and encoding capabilities of torchvision are deprecated ...
```

它来自当前 LeRobot 数据读取链路导入 `torchvision.io`。安装 TorchCodec
只有在数据加载代码实际切换到 TorchCodec backend 后才会生效，并不会自动
消除本目录脚本的所有耗时。`visualize_frame_extras.py` 的整集渲染还包含
逐帧数据读取、PCA/KMeans、overlay 和视频编码，因此长 episode 仍可能较慢。
调试时优先使用 `--max-frames`、较小的 `--panel-size` 和
`--no-contours`。

### 找不到训练配置或 dataset stats

checkpoint 原目录中的训练 `config.yaml` 和 `dataset_stats.json` 应尽量
保留。移动 checkpoint 后，通过
`EVALUATION.training_config_path=...` 和
`EVALUATION.dataset_stats_path=...` 显式传入。

### 找不到 `replan_*.pt`

说明指定的 `raw_predictions/` 目录为空、路径不正确，或 episode 在第一次
replan 前就已结束。先检查 `episode_manifest.json` 和推理日志。

### MP4 无法创建

脚本使用 OpenCV 的 `mp4v` writer。若 `VideoWriter` 打不开，请检查当前
OpenCV 是否带视频编码支持；点云脚本的彩色输出还要求系统中存在
`ffmpeg`。在视频编码不可用时，Dream 渲染产生的逐帧 PNG 仍可单独查看。

### 可视化颜色随帧变化

Dream episode 渲染已经对整个 episode 使用统一的 depth 范围、DINO PCA
和 SAM KMeans。自行调用底层接口时也应复用同一个 projection，而不是每帧
重新拟合。

## Router 阶段特征可视化：rollout 上排，gate 下排

使用训练过新 16-group Router 的 checkpoint，运行一个 LIBERO episode：

```bash
conda activate fastwam
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=src:. \
python Visualize/eval_router_episode.py \
  ckpt=/path/to/new_router_checkpoint.pt \
  EVALUATION.task_suite_name=libero_10 \
  EVALUATION.task_id=0 \
  EVALUATION.initial_state_index=0 \
  EVALUATION.num_inference_steps=10 \
  EVALUATION.output_dir=./evaluate_results/router_demo
```

自动读取 checkpoint 附近的训练配置和 dataset stats，保留其 Dream 去噪步数。
`EVALUATION.dream_inference_steps=8` 可显式覆盖 Dream 步数。
旧版 `router.mode=none` 且没有 `routing_mode` 的 checkpoint 没有这些 gates，
新脚本会明确报错，不会为演示生成虚构 gate。请使用新 Router 训练的权重。

每列对应一次 replan：上方是真实观测（默认主相机），下方是 DINO、CoTracker、
SAM、Depth 四种模态的霓虹色条和原始数值。每种模态对两路相机、两个未来时刻的
四个 group gates 直接取平均。所有 replan 共用固定 `[-1,1]` QK 缩放尺度（负值条向左、正值条向右），不做 softmax、
逐帧归一化、阶段标签或 token-level 可视化。列标题包含 replan 编号及真实环境步。

输出目录：

```text
router_demo/
├── gate_visualization/
│   ├── timeline.png                 # 全部 replans，始终为两排
│   ├── pages/page_000.png            # 默认每页 8 列，便于阅读长任务
│   ├── frames/replan_0000.png        # 单次 replan 的观测和 gates
│   ├── gate_rollout.mp4              # 每次 replan 一帧，默认 5 fps
│   ├── replan_modality_history.npy   # [K,4]：DINO, CoTracker, SAM, Depth
│   └── render_manifest.json         # 数值、环境步、顺序和颜色尺度说明
├── dream_visualization/
│   ├── frames/replan_0000.png        # Dream 预测单独成图，不混入 gate 图
│   └── dream_predictions.mp4
├── raw_predictions/replan_0000.pt    # RGB、完整 Dream tensors、16 gates 和 mapping
├── gates/                           # 原有 [T,16]/[T,4] 和 replan history
├── rollout/                         # 原有观测视频
└── episode_manifest.json            # success、任务、初态、时间步等
```

MP4 是 replan 采样的展示视频，不表示 simulator 实时播放速度。Dream 图继续使用
现有可视化方法：Depth 色图、Tracker 动态掩码、DINO 特征投影、SAM 特征聚类/PCA，
不是把特征预测当作生成 RGB。所有原始预测 tensor 都保留，可以之后重绘。

可选覆盖项：

```bash
GATE_VISUALIZATION.camera=both                # image / wrist_image / both
GATE_VISUALIZATION.columns_per_page=8
GATE_VISUALIZATION.column_width=320
GATE_VISUALIZATION.image_height=224
GATE_VISUALIZATION.fps=5
GATE_VISUALIZATION.save_video=false
GATE_VISUALIZATION.render_dream_images=false  # 跳过单独 Dream 渲染，仍保存原始预测
```

已经保存过带 routing 的 raw records，可以不加载模型和 LIBERO，直接重绘：

```bash
PYTHONPATH=src:. python Visualize/router_gate_visualization.py \
  --raw-dir ./evaluate_results/router_demo/raw_predictions \
  --output-dir ./evaluate_results/router_demo/gate_visualization \
  --columns-per-page 8 --camera image --fps 5
```

每次新测评请选择新的 output_dir，避免不同 episode 的记录混合。
