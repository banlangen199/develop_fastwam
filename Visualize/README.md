# FastWAM 可视化工具

`Visualize/` 中的工具分为两类：

- **在线预测可视化**：运行一个 LIBERO episode，保存 DreamFastWAM 在推理时预测的未来模态，再统一渲染。
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

1. 正常预测一段 action；
2. 只在该 action chunk 的第一次 denoising 前向中解码 Dream；
3. 将原始预测先保存到 CPU；
4. episode 完成后统一拟合颜色映射并生成可视化。

这样不会在每个 denoising step 重复运行 Dream decoder，也不会在 rollout
过程中逐帧进行 PCA、KMeans 和视频编码。

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
| depth | `[128, 256]` | 64 个输出位置，每个位置还原 `16×16` depth patch |
| dyn | `[392, 1]` | `14×14` 动态网格 |
| DINO | `[16, 32, 768]` | `16×16×768` |
| SAM | `[16, 32, 256]` | `16×16×256` |

其中 DINO 和 SAM 沿宽度方向拼接；depth 和 dyn 先以两个视角的扁平位置
保存，再由渲染器拆分。第一半是主视角 `image`，第二半是腕部视角
`wrist_image`。

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
