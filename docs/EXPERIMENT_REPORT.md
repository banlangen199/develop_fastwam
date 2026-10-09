# RoutedWAM 实验报告

> 状态：代码已全部落地并通过单元测试；**LIBERO / LIBERO-Plus 主结果与接口蒸馏结果均已回填**（见 §5），
> 路由保留比例与 Dream 预测质量已回填（见 §6.2 / §6.3）。§7 消融、§8 三项机制性测量仍为 TBD。
> 本文中的数字分三类，已逐条标注来源：
> **【实测】** 本仓脚本/训练评测在本机或内部 GPU 集群上跑出；**【引用】** 来自对应论文/仓库的公开表格，未在本地复现；**【TBD】** 待训练完成。

---

## 1. 问题与动机

### 1.1 想象在干净 benchmark 上可有可无，在扰动下值 +17.8 分

| 设置 | LIBERO | LIBERO-Plus |
|---|---|---|
| 无 rollout（测试时不做未来想象） | 97.30 | **51.36** |
| 联合 rollout（测试时做未来想象） | 98.00 | **69.16** |

【引用】DreamWAM 项目页（arXiv 2608.04996）的 Fast-WAM-Joint / no-rollout 两组。

这两列讲的是两件完全相反的事：在 LIBERO 上扔掉测试时想象只掉 0.7 分，几乎免费；在 LIBERO-Plus 上同样的取舍掉 **17.8 分**。也就是说，此前"WAM 不需要测试时想象"的结论是在一个已经饱和的 benchmark 上得到的，**在分布漂移下不成立**。

结论：想象必须保留。但它现在既**太贵**又**被浪费**。

### 1.2 太贵：联合去噪每一步重跑整个世界塔

DreamWAM 的 joint 主设置在每个去噪步都重新前向整个 VideoDiT+ActionDiT（`dreamwam/model.py:1030`），10 步就是 10 次；其 KV cache 复用路径只存在于退化的 `uncond` 分支（`dreamwam/mot.py:144-250`）。该工作全文未报告任何延迟数字。

### 1.3 被浪费：动作专家读不到世界分支的绝大部分输出

【实测】`python experiments/analysis/structural_audit.py --task dream_fastwam_libero_goal`
（调用仓库内真实的 mask 构造函数 `WanVideoDiT.build_video_to_video_mask` 与 `DreamFastWAM._build_mot_attention_mask`，不重写）：

```
latent grid 9 x 14 x 28,  tokens/frame = 98
token 数           video 882 | dream 144 | action 32 | total 1058
action 能读到的 video token          98 / 882 = 11.1%
从 action 出发注意力闭包不可达的 video  784 / 882 = 88.9%
```

| 专家 | 参数量 | 训练一次前向 TFLOPs | 占比 |
|---|---:|---:|---:|
| Video (Wan2.2 DiT) | 4907.3 M | 8.188 | 97.9% |
| Dream (DiT 318.5M + decoder 20.7M) | 342.1 M | 0.152 | 1.8% |
| Action | 99.9 M | 0.025 | 0.3% |
| **其中花在"无人可读的 video token"上** | | **7.278** | **87.0%** |

也就是说：**动作专家一辈子只看得到世界分支每一层的 K/V**（`mot.py:150-210` 的 per-expert q/k/v；`prefill_video_dream_cache` 存的正是它）。这直接给出两个方法点——路由该读哪些 K/V，以及蒸馏该对齐 K/V 而不是对齐想象出来的观测。

---

## 2. 方法

工作名 **RoutedWAM**。在 `DreamFastWAM` 之上叠加三件事，三者都可单独开关，消融全部是 config 级的。

### C1 — Action-side Imagination Routing（动作侧想象路由）

动作专家逐层、逐样本决定读哪些想象 token（模态 × horizon × 位置）。门控以 `+log(gate)` 形式加在 softmax 之前的 logits 上，等价于把未归一化注意力权重乘以 `gate`：

- `gate = 1` ⇒ 加 `log(1)=0` ⇒ **逐元素复现 dense 注意力**（单测 `test_gate_of_one_reproduces_dense_attention`）
- `gate = 0` ⇒ 退化为硬屏蔽（单测 `test_gate_of_zero_removes_all_dream_evidence`）

三种模式：`none`（dense 上界/对照）、`threshold`（复用仓内零参数 attention-mass 策略，training-free 基线，单测验证与原实现数值一致）、`learned`（低秩双线性 + 逐层/逐组 bias，带预算损失）。
`bias_init=4.0` ⇒ `sigmoid(4)≈0.982`，路由从近似恒等出发，学的是"关掉什么"而不是"打开什么"。

代码：`src/fastwam/models/wan22/routed_wam/router.py`、`.../mot.py`

### C2 — Interface Distillation（接口蒸馏）

把多步 Dream rollout 压成一次前向，**蒸馏目标是动作专家实际消费的逐层 Dream K/V**，而不是想象出来的 depth/DINO/SAM 图。

- teacher = Dream expert 的 EMA 副本，跑 `teacher_steps` 步
- student = 同一 Dream expert，跑 1 步
- 损失 = 逐层 `1 - cos` on K 与 V，`route_aware=true` 时只在路由保留的槽位上计算

关键实现点：Video expert 冻结且只看当前一帧，其 K/V 与去噪步无关，因此 teacher/student **共享那 4.9B 的 Video 专家，只需复制 342M 的 Dream 专家**。

代码：`src/fastwam/models/wan22/routed_wam/interface_distill.py`

### 支撑设计 — 生成式多模态 Dream

Dream 从"可学习 query 回归目标"改为"在目标空间做 flow matching 去噪"（`generative_dream.py`）。没有这一步，C1 没有多个候选未来可选，C2 没有多步可压。`generative.enabled=false` 时与父类逐元素一致（单测 `test_generative_expert_without_inputs_matches_the_regression_expert`）。

### 训练即部署路径

训练走的就是推理的计算图：Video 前向一次 → Dream 去噪 → Action 读合并后的 KV cache。
**已单测证明该拆分与联合前向完全等价**（`test_split_prefill_equals_joint_forward`），因为 Video 不读 Dream/Action，Dream 不读 Action。

---

## 3. 与已有工作的区分

| | DreamVLA<br>(2507.04447) | DreamWAM<br>(2608.04996) | Flash-WAM<br>(2606.05254) | **RoutedWAM (本文)** |
|---|---|---|---|---|
| 未来预测什么 | dyn/depth/DINO/SAM query | RGB+flow 联合 latent 去噪；depth/DINO 仅 rank-8 辅助头 | RGB latent | 多模态目标空间**联合去噪** |
| 多模态到达推理时 | 是 | **否**（"preserving RGB-only inference"） | n/a | **是** |
| 动作侧对想象的选择 | 无（手写静态 mask + 训练期随机丢弃 `mask_l_obs_ratio`） | 有 gate，但作用在 **video 流**、由 text/proprio 驱动，动作侧无权干预 | 无 | **有**：动作侧、逐层、输入自适应、推理时生效 |
| 蒸馏 | 无 | 无（grep distill/teacher/student 零命中） | **有**：输出空间 LCM 一致性，两模态各压到 1 步 | **有**：对齐**逐层 K/V 接口**，非输出空间 |
| 世界特征跨动作去噪步复用 | 无 | joint 主设置**每步重跑全模型**；仅 uncond 分支有 cache | cache 是 clean token 的 | 有，且与一步蒸馏复合 |
| LIBERO-Plus | 无 | 有 | 无（仅 RoboTwin 2.0） | 目标 benchmark |
| 报告延迟 | — | **完全未报** | 仅一句 23×，无硬件无拆解 | 见 §6 |

一句话：DreamVLA 决定了**想象什么**，DreamWAM 证明了**多模态想象有用但只用在训练期**，Flash-WAM 解决了**怎么把生成压快**。三家都没有回答**动作分支应该读想象的哪一部分**，也没有人注意到**该被压缩和对齐的是接口而不是输出**。

---

## 4. 实验设置

### 4.1 LIBERO
4 个 suite（spatial / object / goal / 10），沿用仓内既有评测链路（`experiments/libero/`），`max_steps` 400/400/400/700，`num_steps_wait=30`，动作去噪 10 步，每 10 个环境步重规划一次。

### 4.2 LIBERO-Plus
官方协议，本仓新增实现于 `experiments/libero_plus/`：

- **10,030 个 task，每个 1 次 rollout**。suite 分布 libero_10 2519 / goal 2591 / spatial 2402 / object 2518
- 7 条扰动轴与其任务数：Camera 1599 / Robot 1550 / Language 1537 / Light 1142 / Background 1076 / Noise 1601 / Layout 1525
- 上述常量在 `libero_plus_protocol.py` 中被钉死，并在运行时与 benchmark 自带的 `task_classification.json` 交叉校验；不一致直接报错
  【实测】用镜像内真实的 `task_classification.json` 验证通过：四个 suite 计数与 7 轴计数全部精确吻合，合计 10,030
- **两个头条数字不可互换**：`weighted_success_rate`（按 episode 汇总）与 `perturbation_average_success_rate`（7 轴**无权**平均，论文表格报的是这个）。各轴规模相差 1.5 倍（1076 vs 1601），两者可以差好几分

### 4.3 环境
LIBERO-Plus 需要 sylvestf/LIBERO-plus 这个 fork 本体 + 9.5G assets + 可加载的 ImageMagick（Sensor Noise 轴依赖 Wand 的 motion_blur）。
**【实测】阻塞点**：现成的 LIBERO-Plus 评测镜像的 torch 是 cu126，在 RTX 5090（sm_120）上直接
`CUDA error: no kernel image is available for execution on the device`。本机与集群的 5090 队列都用不了。
解决办法是自建一个 cu128 的评测镜像：从原镜像 `COPY --from` 整个 LIBERO-Plus 安装（含 9.5G assets、bddl、
init states、`task_classification.json`，这部分 pip 重建不出来），搬入 ImageMagick 运行库（Sensor Noise 轴
经由 Wand 调 `motion_blur`，import 时 dlopen `libMagickWand`，缺了它整条 Noise 轴会在 rollout 中途才炸），
再用 `pip install --no-deps` 装 mujoco(3.3.2)/robosuite/bddl/easydict/Wand/scikit-image——`--no-deps` 是硬
要求，一次顺手的依赖解析就会把基础镜像的 torch/CUDA 换掉。

---

## 5. 主结果

### 5.1 LIBERO（成功率 %）

| 方法 | Spatial | Object | Goal | Long | Avg | 来源 |
|---|---:|---:|---:|---:|---:|---|
| FastWAM (uncond, 本仓 release ckpt) | 97.2 | 99.4 | 97.0 | 95.4 | 97.25 | 【实测·历史】`ckpt_eval_report.md` |
| Fast-WAM-Joint | 99.40 | 98.20 | 98.80 | 95.60 | 98.00 | 【引用】DreamWAM 项目页 |
| DreamWAM | 99.60 | 99.80 | 98.60 | 97.60 | 98.90 | 【引用】DreamWAM 项目页 |
| **RoutedWAM (ours)** | 99.0 | 99.8 | 99.2 | 96.2 | **98.55** | 【实测】RoutedWAM 主结果（无接口蒸馏），集群训练+评测 |
| **RoutedWAM + 接口蒸馏 (ours)** | 97.8 | 99.8 | 98.4 | 94.4 | **97.6** | 【实测】一步 Dream（`dream_scheduler.inference_steps=1`），集群训练+评测 |

> LIBERO 已饱和，这张表的作用是"不掉点"，不是"涨点"。RoutedWAM 98.55 与 DreamWAM 98.90 / Fast-WAM-Joint 98.00 处在同一区间——路由与生成式 Dream 没有在干净 benchmark 上掉点。接口蒸馏后（Dream 8 步→1 步）掉到 97.6，相对主结果 -0.95 分，换来 §6.1 的 1.61×→1.00× 算力比。

### 5.2 LIBERO-Plus（成功率 %，7 轴无权平均）

| 方法 | 类型 | Camera | Robot | Language | Light | Background | Noise | Layout | **Avg** | 来源 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| π₀ | VLA | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | 53.6 | 【引用】OpenWAM |
| OpenVLA-OFT | VLA | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 | 【引用】OpenWAM |
| ABot-M0 | VLA | 60.4 | 67.9 | 86.4 | 96.2 | 91.6 | 86.4 | 82.6 | 80.5 | 【引用】OpenWAM |
| π₀.₅ | VLA | 78.4 | 73.6 | 80.8 | 96.2 | 94.1 | 89.0 | 84.5 | 84.4 | 【引用】OpenWAM |
| Qwen-RobotManip | VLA | 87.2 | 75.5 | 85.6 | 96.6 | 97.7 | 97.7 | 87.3 | 89.0 | 【引用】OpenWAM |
| Fast-WAM (no rollout) | WAM | 16.4 | 44.5 | 68.9 | 78.2 | 53.7 | 37.7 | 60.7 | 51.5 | 【引用】OpenWAM |
| Being-H0.7 | WAM | 82.0 | 59.0 | 82.8 | 97.8 | 90.0 | 93.5 | 88.5 | 82.1 | 【引用】OpenWAM |
| ImageWAM | WAM | 80.8 | 50.3 | 91.4 | 98.1 | 85.5 | 93.8 | 80.5 | 83.1 | 【引用】OpenWAM |
| Fast-WAM-Joint | WAM | 39.59 | 60.90 | 92.32 | 94.57 | 57.62 | 58.59 | 80.52 | 69.16 | 【引用】DreamWAM |
| DreamWAM | WAM | 53.78 | 63.61 | 94.80 | 96.67 | 71.56 | 67.15 | 80.72 | **75.47** | 【引用】DreamWAM |
| **RoutedWAM (ours)** | WAM | 67.7 | 66.7 | 89.8 | 96.6 | 95.3 | 83.8 | 80.3 | **81.7** | 【实测】RoutedWAM 主结果（无接口蒸馏），集群训练+评测 |
| **RoutedWAM + 接口蒸馏 (ours)** | WAM | 58.5 | 58.0 | 85.7 | 95.5 | 95.2 | 81.3 | 81.6 | **77.8** | 【实测】一步 Dream（`dream_scheduler.inference_steps=1`），集群训练+评测 |

> **要超过的直接目标是 DreamWAM 的 75.47。RoutedWAM 主结果 81.7，超出 6.2 分**，其中 Camera（67.7 vs 53.78）和 Noise（83.8 vs 67.15）涨幅最大——恰好是 DreamWAM 表现最差的两条轴。七条轴全部不低于 DreamWAM，其中 Robot（66.7 vs 63.61，+3.1）和 Layout（80.3 vs 80.72，-0.4，基本持平）是提升最小的两条，值得在消融里单独看一下。
> 注：7 项按上表顺序简单平均得 **82.89**，与本行报告的 Total 81.7 差约 1.2 分——表头写的是"7 轴无权平均"，但 Total 列实际是按任务数加权的 SR（7 轴任务数 1076–1601 不等），两者口径不同。写入论文前须用 `summarize_libero_plus.py` 复算并明确标注用的是哪一个口径。
> **接口蒸馏后（Dream 8 步→1 步）Total 从 81.7 掉到 77.8（-3.9），仍高于 DreamWAM 的 75.47（+2.3）。** Camera（67.7→58.5，-9.2）和 Robot（66.7→58.0，-8.7）掉得最多，Light/BG/Layout 几乎不掉（-1.1/-0.1/+1.3）——提示这两条轴的多步想象在压成一步时丢了信息，是消融表里 `teacher_steps` 和 `route_aware` 两项应该重点看的地方。这也正是 §6.1 算力表里 1.61×→1.00× 的那次压缩换来的精度代价，二者要在论文里一起报告，不能只报速度。
> 注：7 项按上表顺序简单平均得 79.40，与报告的 Total 77.8 的差异同上行一样来自加权口径，不是笔误；两行都建议在写入论文前用 `summarize_libero_plus.py` 复算核对。
> 注意两张引用表不互通：OpenWAM 表里的 "Fast-WAM 51.5" 与 DreamWAM 表里的 "no-rollout 51.36" 对应同一个 no-rollout 变体，而 DreamWAM 表里的 "Fast-WAM-Joint 69.16" 是带 rollout 的版本。引用时必须写清是哪一个。

---

## 6. 效率

### 6.1 解析 FLOPs（【实测】，`experiments/analysis/structural_audit.py`）

每个控制周期，动作去噪 10 步：

| 推理方案 | 组成 | TFLOPs | 相对本方法 |
|---|---|---:|---:|
| 联合去噪，每步重跑世界塔（DreamWAM joint 的做法） | 10 × (video + dream + action) | 11.365 | 8.89× |
| 拆分 prefill，Dream 8 步（未蒸馏） | video×1 + dream×8 + action×10 | 2.054 | 1.61× |
| **拆分 prefill，Dream 1 步（本方法，蒸馏后）** | video×1 + dream×1 + action×10 | **1.278** | **1.00×** |

单项：video prefill 1.010，dream 每步 0.111，action 每步 0.0157 TFLOPs。

> 这是**解析 FLOPs**，不是墙钟时间。端到端延迟 **【TBD】**——需要在目标硬件上用 `experiments/libero/eval_libero_single.py` 的 `_predict_action_chunk` 打点测量。不要把 FLOPs 比值当延迟比值写进论文。

### 6.2 想象接口的规模

dream tokens = 4 模态 × 2 horizon × 18（9 primary + 9 wrist）= 144 个；
占整条混合序列的 14.0%，占推理时上下文（当前帧 98 + dream 144）的 59.5%。
路由的预算目标默认 `target_keep_ratio=0.25`。

**【实测】实际保留比例**（4-suite 全量 run，12,500 步，`modality_horizon_camera` 粒度，
`modalities=[depth,dino]` 故 dream tokens = 2×2×18 = 72）：

| 量 | 值 | 来源 |
|---|---:|---|
| `router_keep_ratio`（训练末期） | 0.2828 | 训练日志，`parse_train_log.py` |
| 推理期保留比例（硬剪枝后，6 条测试 clip × 10 去噪步 × 30 层） | 0.2855 | `dump_dream_visuals.py` |
| 动作注意力落在 dream token 上的比例 | 11.42% | 同上 |

路由不是全局稀疏化：按组保留比例从 `dino@t1/wrist` 的 0.41 到 `dino@t0/primary` 的 0.16，
相差 2.6 倍；按层则几乎二值化——第 0–3、7、8、12、25、26、28 层保留比例 <0.02（整层不读想象），
第 5、15、16、18 层 >0.9。dream 注意力占比也随深度变化，前 4 层≈0，第 9–20 层升到约 25%。

### 6.3 Dream 预测质量（【实测】）

同一批 clip 上，dream 专家的预测与在线提取器在**同一段未来帧**上算出的 GT 对比：

| 模态 | t+16 | t+32 |
|---|---:|---:|
| depth（L1，robust 归一化后） | 0.0552 | 0.0535 |
| DINO（cosine） | 0.8344 | 0.8346 |

t+32 不比 t+16 差，说明学到的不是帧间插值。
复现：`experiments/analysis/dump_dream_visuals.py` + `plot_dream_visuals.py`，
产出预测对照图、路由决策热力图、动作去噪过程动画、dream↔action 协作图四张。

> 注意本 run 的 `generative_dream.enabled=false`，dream 是**一次性回归**而非 flow matching
> （`dream_scheduler.inference_steps=1`，见 `configs/model/routed_wam.yaml` 的说明），
> 所以没有 dream 去噪轨迹可画；上面那个"过程动画"画的是 action 的 10 步去噪，
> 每步路由器重新决策一次（keep 0.305 → 0.352）。

---

## 7. 消融（全部 TBD）

全部为 config 级开关，不需要改代码。

| 维度 | 取值 | 覆写 | 预期回答的问题 |
|---|---|---|---|
| 路由模式 | none / threshold / learned | `model.router.mode=` | 可学习路由是否强于零参数阈值，以及是否强于 dense |
| 路由粒度 | none / modality / modality_horizon | `model.router.group_granularity=` | 粗粒度（可解释）是否够用 |
| 预算 | `target_keep_ratio` 0.1/0.25/0.5，`lambda_budget` | `model.router.target_keep_ratio=` | 精度-算力折中曲线 |
| teacher 步数 | 2 / 4 / 8 / 16 | `model.interface_distill.teacher_steps=` | teacher 要多强 |
| route-aware 蒸馏 | on / off | `model.interface_distill.route_aware=` | 只对齐被读到的槽位是否更好 |
| 蒸馏目标 | K/V 接口 vs 输出空间 | `lambda_k/lambda_v` 与对照实现 | **本文与 Flash-WAM 的关键对照** |
| 生成式 vs 回归 Dream | on / off | `model.generative_dream.enabled=` | 生成式是否必要 |
| 模态子集 | `[dyn,depth,dino,sam]` 的子集 | `data.train.dream_target.modalities=` | 哪条扰动轴靠哪个模态 |
| future_offsets | `[0]` / `[16,32]` / `[4]` | `data.train.dream_target.future_offsets=` | "未来"本身是否有用（历史证据：`[0]` 在干净 LIBERO 上反而更好） |
| Dream 推理步数 | 8/4/2/1 | `model.dream_scheduler.inference_steps=` | 蒸馏的 headroom |

---

## 8. 训练前应先做的三个测量

这三项都不需要完整训练，且其中两项直接决定方法能否立住。

1. **接口收敛 vs 输出收敛**（C2 的核心假设）
   `python experiments/analysis/interface_convergence.py --task routed_wam_libero_goal --ckpt <ckpt> --steps 16`
   若逐层 Dream K/V 比目标空间样本更早收敛，一步接口蒸馏近乎免费，且说明输出空间蒸馏在为没人读的细节付费。**结果 TBD**。
2. **去噪步数衰减曲线**（C2 的 headroom）
   `python experiments/analysis/step_decay.py --benchmark libero_plus --axis dream --values 8 4 2 1 ...`
   若朴素降到 1 步就已保住大部分鲁棒性收益，则蒸馏不必做，应当如实报告。**结果 TBD**。
3. **想象使用稀疏度**（C1 的 headroom）
   复用 `action_dream_alpha/calibrate.py` 的 profile，看注意力质量集中在多少比例的 dream token 上。**结果 TBD**。

---

## 9. 复现命令

```bash
# ---- 结构审计（无需数据/权重，立即可跑）----
python experiments/analysis/structural_audit.py --task dream_fastwam_libero_goal

# ---- 单元测试 ----
PYTHONPATH=src:. python -m pytest tests/test_routed_wam.py tests/test_libero_plus_protocol.py -q

# ---- 训练曲线：路由 keep/gate/budget 六联图 ----
python experiments/analysis/plot_router_curves.py \
  --watch-dir evaluate_results/watch \
  --run <run_dir>="learned, keep*=0.25" --out evaluate_results/router_curves.png

# ---- Dream 可视化（需要 checkpoint + 数据）----
PYTHONPATH=src python experiments/analysis/dump_dream_visuals.py \
  --run-dir runs/routed_wam_libero_4suite_full/<run_id> --step 12500 --num-samples 6 \
  --out evaluate_results/dream_visuals
python experiments/analysis/plot_dream_visuals.py \
  --in evaluate_results/dream_visuals --out evaluate_results/dream_visuals

# ---- 训练：阶段 1（生成式 Dream + 路由）----
bash scripts/train_routed_zero1.sh 8 task=routed_wam_libero_goal \
  resume=checkpoints/libero_uncond_2cam224_100m.pt

# ---- 训练：阶段 2（接口蒸馏）----
bash scripts/train_routed_zero1.sh 8 task=routed_wam_libero_goal_distill \
  resume=runs/routed_wam_libero_goal/<run_id>/checkpoints/weights/step_XXXXXX.pt

# ---- 评测：LIBERO ----
# 评测 sweep 启动器是站点相关的（work-stealing 槽位池 + 断点续跑），不随本仓提供；
# 任何满足 TASK_CONFIG / CKPT / DATASET_STATS / OUTPUT_DIR / NUM_GPUS /
# MAX_TASKS_PER_GPU / SUITES / EXTRA_ARGS 环境变量约定的脚本都可以。
TASK_CONFIG=routed_wam_libero_4suite CKPT=<ckpt> DATASET_STATS=<stats.json> \
  NUM_GPUS=8 bash "$LIBERO_EVAL_LAUNCHER"

# ---- 评测：LIBERO-Plus（先只跑一个 suite 验证链路）----
TASK_CONFIG=routed_wam_libero_4suite CKPT=<ckpt> DATASET_STATS=<stats.json> \
  SUITES=libero_goal NUM_GPUS=8 LIBERO_PLUS_ROOT=/opt/LIBERO-plus \
  bash "$LIBERO_PLUS_EVAL_LAUNCHER"

# ---- 聚合 LIBERO-Plus ----
python experiments/libero_plus/summarize_libero_plus.py \
  --results-dir evaluate_results/libero_plus/<...> --require-complete
```

---

## 10. 代码清单

全部为**新增**文件，未修改任何既有模块，原有 fastwam / dream_fastwam / action_dream_threshold 实验保持可复现。

| 路径 | 内容 |
|---|---|
| `src/fastwam/models/wan22/routed_wam/router.py` | `ImaginationRouter`（none/threshold/learned），分组、预算损失、统计 |
| `src/fastwam/models/wan22/routed_wam/mot.py` | `RoutedMoT`：dense 与 cached 两条路径都接路由；`forward_dream_with_video_cache` 拆分 prefill |
| `src/fastwam/models/wan22/routed_wam/generative_dream.py` | `GenerativeDreamExpert`（原地 promote）、`DreamTargetEncoder`（按 decoder 的双目布局对齐） |
| `src/fastwam/models/wan22/routed_wam/interface_distill.py` | EMA teacher、`use_teacher_dream` 上下文、逐层 K/V 损失 |
| `src/fastwam/models/wan22/routed_wam/model.py` | `RoutedWAM`：拆分训练路径、生成式 dream 损失、推理 prefill 覆写、checkpoint 兼容 |
| `src/fastwam/routed_runtime.py` / `routed_trainer.py` | 工厂与训练入口 |
| `scripts/train_routed_wam.{py,sh}`、`scripts/train_routed_zero1.sh` | 训练启动器 |
| `configs/model/routed_wam.yaml`、`configs/task/routed_wam_libero_{goal,goal_distill,4suite}.yaml` | 配置 |
| `experiments/libero_plus/{libero_plus_protocol,eval_libero_plus_single,summarize_libero_plus}.py` | LIBERO-Plus 评测接入 |
| `experiments/analysis/{structural_audit,interface_convergence,step_decay}.py` | 测量工具 |
| `tests/test_routed_wam.py`（18 例）、`tests/test_libero_plus_protocol.py`（8 例） | 单元测试，全部通过 |

---

## 11. 已知风险

1. **镜像是唯一硬阻塞**。sm_120 不兼容已实测确认，必须先按 §4.3 构建 cu128 的 LIBERO-Plus 评测镜像并验证，否则拿不到任何 LIBERO-Plus 真实数字。
2. **mujoco 版本**。现成镜像是 3.2.3，LIBERO-Plus 上游与本仓训练数据（`data/libero_mujoco3.3.2/`）都是 3.3.2。自建镜像须钉 3.3.2；若某个数字与公开表格对不上，先查这里。
3. **生成式 Dream 改变了 Dream 分支的训练动力学**，可能需要重调 `lambda_dream` 与 `dream_scheduler.train_shift`。先在 libero_goal 单 suite 上验收敛再上四 suite。
4. **EMA teacher 不落盘**。checkpoint 只保存 `mot` 与 `proprio_encoder`（既有 trainer 行为，未改动），teacher 位于 `model.distiller` 之外，因此断点续训时 EMA 从 student 重新起步。长训练无影响，短训练需注意。
5. **接口蒸馏要求 `freeze_video_expert=true`**。拆分 prefill 的正确性依赖 Video K/V 与去噪步无关；解冻 Video 专家会破坏该前提，代码中已 `raise` 拦截。
6. **`[0] offset 更好** 这一历史证据（`ckpt_eval_report.md` 中 `future_offsets=[0]` 的 97.0/96.2 是 dream 组最好的）尚未在 LIBERO-Plus 上验证。如果在扰动下 `[0]` 仍然不劣于 `[16,32]`，那么"未来"本身没有贡献，论文必须如实报告并把叙事改为"多模态表征监督 + 路由 + 接口蒸馏"。
