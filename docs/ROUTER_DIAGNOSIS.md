# 路由诊断：筛选结果与训练曲线

生成：`experiments/analysis/plot_router_curves.py`
数据：`evaluate_results/router_curves.csv`（41 个记录步 × 5 个 run）
图：`evaluate_results/router_curves.png`

六个 run 全部 `Succeeded`（libero_goal，416 步，2×8 GPU，10/01 提交）。

| cell | 配置 |
|---|---|
| E1 | learned, λ_budget=0.01, keep\*=0.25 |
| E2 | learned, λ_budget=0.1 |
| E3 | learned, λ_budget=0.1, keep\*=0.10 |
| E4 | threshold（零参数对照） |
| E5 | none（dense 上界） |
| E6 | learned, future_offsets=[0] |

---

## 1. 师弟的两个现象：**同时出现**才是 bug，单独出现是正常的

### 1.1 `loss_router_budget=0.0000` 本身有两种完全不同的来源

预算项的定义（`router.py:407`）：

```python
if not gates or self.config.lambda_budget <= 0.0 or self.config.mode != "learned":
    return torch.zeros((), device=...)          # ← 来源 A：gates 为空，恒等于 0
mean_gate = torch.stack([g.mean() for g in gates]).mean()
return float(self.config.lambda_budget) * (mean_gate - target).pow(2)   # ← 来源 B
```

λ=0.01、keep\*=0.25。代入实测的 `router_gate_mean`：

| gate | 真实 budget loss | 4 位小数打印 |
|---|---|---|
| 0.9958（step 10） | 0.01·0.7458² = **5.56e-3** | `0.0056` |
| 0.2913（step 416） | 0.01·0.0413² = **1.7e-5** | `0.0000` |

**来源 B**：gate 收敛到 target 后这一项掉到 1e-5 量级，4 位小数下必然打印成 `0.0000`。这是「已达标」，不是故障。图里第 3 个 panel 用对数轴正是为了让这个地板显示成下降曲线而不是消失。

### 1.2 但师弟的两个现象**不能同时**由来源 B 解释

这是判定的关键：

- 若 gate ≈ 1 → budget = 0.01·(1−0.25)² = **0.0056**，不可能是 `0.0000`
- 若 budget = `0.0000` → gate 必须 ≈ 0.25，不可能「收敛到 1」

两者互斥。**同时成立的唯一解释是来源 A：`gates` 列表为空** → `budget_loss` 精确返回 0 → gate 拿不到任何剪枝梯度 → 永久停在初始化值 `sigmoid(bias_init) = sigmoid(4) = 0.982 ≈ 1`。

这正是 `mot.py:397` 修的那个 bug：split 训练路径走 `_action_attention_with_context_cache`，而 gate 当时只在 `forward` 里被收集。已用反向验证确认该行是 load-bearing（注释掉 → `test_cached_path_collects_gates_so_the_budget_loss_is_live` 失败）。

### 1.3 另外，前 42 步「看起来没动」是设计使然

- `bias_init: 4.0` → gate **故意从 0.982 起步**（学「关掉什么」而非「打开什么」）
- `warmup_ratio: 0.1` → 416×0.1 ≈ **前 42 步 keep_ratio 恒等于 1.0000**

所以前 ~40 步的健康日志也长这样：`gate≈0.99, keep_ratio=1.0000`。第 50 步起 E1 的 keep_ratio 掉到 0.72，最终稳定在 **0.653**。

**给师弟的判据** —— 不要只看 `loss_router_budget`：

| | gate | keep_ratio | budget |
|---|---|---|---|
| 真的死了（来源 A） | 永久钉在 ≈0.98 | 永久 1.000 | 永久 `0.0000` |
| 正常收敛（来源 B） | 落到 0.25~0.31 | 0.55~0.65 | 先 `0.0056` 再 `0.0000` |
| 还在 warmup | ≈0.99 | 1.000 | `0.0056` |

---

## 2. 筛选结果

### 2.1 全局保留率（最终步）

| cell | gate | keep_ratio | 保留 token 数 /72 | loss_action |
|---|---|---|---|---|
| E1 learned λ=.01 | 0.310 | 0.653 | 47.0 | 0.100 |
| E2 learned λ=.1 | 0.178 | 0.191 | 13.7 | 0.120 |
| E3 learned λ=.1 keep\*=.10 | 0.006 | 0.010 | 0.7 | 0.123 |
| E4 threshold | 0.671 | 0.671 | 48.3 | 0.105 |
| E5 none | — | 1.000 | 72 | 0.106 |
| E6 learned offsets=[0] | 0.310 | 0.557 | 20.0 | 0.100 |

λ_budget 单调控制稀疏度（0.653 → 0.191 → 0.010），**预算机制是有效的**。

E3 几乎剪光（0.7/72）而 `loss_action` 只从 0.100 涨到 0.123 —— 在 416 步这个尺度上，动作分支对想象 token 的依赖比预期弱。这需要成功率才能定性。

### 2.2 ⚠️ 核心负面结果：learned router 没有「路由」，只是在「稀疏」

`group_granularity=modality_horizon_camera`，8 个组 = {depth,dino}×{t0,t1}×{primary,wrist}。各组最终 keep_ratio 的离散程度：

| cell | 组间 spread | 结论 |
|---|---|---|
| E1 learned λ=.01 | **0.059** | 近乎均匀 |
| E2 learned λ=.1 | **0.011** | 近乎均匀 |
| E6 learned offsets=[0] | **0.020** | 近乎均匀 |
| **E4 threshold（零参数）** | **0.340** | 明显区分 |
| E3 learned λ=.1 keep\*=.10 | 0.020 | 全剪光，无意义 |

E4 的逐组值：`depth@t0/primary 0.795`、`dino@t0/primary 0.489`、`dino@t1/wrist 0.829` —— 零参数的 attention-mass 策略在不同模态/视角/horizon 上的差异是 learned router 的 **6 倍**。

**这对 C1（Action-side Imagination Routing）是不利的**：learned router 当前学到的等价于一个全局稀疏度旋钮，而「逐模态/逐视角自主选择」这一卖点由**不含参数的基线**体现得更好。E3 的 CV% 看起来很高（55.7%）纯粹是因为均值接近 0 导致的除法放大，不能算作选择性。

### 2.3 这不能说明感知阶段假设对错

以上全部是**训练集上、整个 batch 平均**的统计量。用户的假设是「**测试时不同阶段**感知属性不同（早期全局语义 → 接触前局部 3D/腕部）」，这需要 `perception_phase.py` 在 rollout 上按阶段分桶，训练期平均值在原理上无法回答。

同时必须留的口径：**这六组只产出训练 loss，不是成功率。LIBERO / LIBERO-plus 评测一次都没跑过**，报告里的成功率数字全部来自用户提供，不是本仓复现的。

---

## 3. 复现

```bash
# 拉日志（watcher 只有 2/6 存活，其余需手工补）
for lf in $(aidictl job logs list <JOB> | awk '/-main\.log/{print $NF}'); do
    aidictl job logs cat <JOB> "$lf" >> evaluate_results/watch/<JOB>/job_log.txt
done

# 解析成表（注意：指标是换行折叠的，单行 grep 只能抓到落在表头那行的几个）
python experiments/analysis/parse_train_log.py evaluate_results/watch/*/job_log.txt \
    --csv evaluate_results/router_curves.csv

# 出图
python experiments/analysis/plot_router_curves.py
```

`parse_train_log.py` 存在的原因：trainer 把指标**折行**打印在 `>> epoch=.. step=N/TOTAL` 下面，一行一到两个 `key=value`。没有任何一行包含完整的一个 step，所以 `grep 'step=' | awk` 只会静默地抓到落在表头行上的那几个指标 —— 这是我自己之前读错 `loss_router_budget` 的直接原因。
