# reduce_grad

`reduce_grad` 把各 rank 槽位里的副本梯度归并回 owner 的 home 梯度。
`local_grads` 是 GMM 对“home + 副本”全量视图反传得到的 full-dW 形态；
`MoonepBuffer.reduce_grad()` 透传本算子。

## 1. 接口

```python
reduce_grad(
    context, experts_to_copy, local_grads, epn,
    *, slot=0, slots=None, async_finish=False,
)
```

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| `context` | 输入 | MoonEP 通信上下文 | - | - |
| `experts_to_copy` | 输入 | `moonep_plan` 输出的副本清单，shape `(R, epn)`（连续、NPU） | INT32 | ND |
| `local_grads` | 输入 | full-dW 梯度，2 或 3 组，每组 shape `(2*epn, Wg)` 或 `(2*epn, Hp, H)`（前 epn 行 home 种子、后 epn 行副本） | BFLOAT16、FLOAT16、FLOAT32 | ND |
| `epn` | 输入 | 每 rank 本地专家数 `E / R` | INT | - |
| `slot` | 可选输入 | 窗口槽位，仅校验与 `prefetch_weight` 一致，缺省 0 | INT | - |
| `slots` | 可选输入 | 窗口槽数；None 取构造定型值 | INT（可为 None） | - |
| `async_finish` | 可选输入 | 异步完成开关，缺省 False | BOOL | - |
| `home_views` | 输出 | 原位累加结果，各组 shape `(epn, Wg)` | 同 `local_grads` | ND |
| `event` | 可选输出 | `async_finish=True` 时追加的发起流事件 | - | - |

## 2. 接口约束

- `local_grads` 是 2 或 3 组 full-dW 张量（GMM 对全量视图反传的形态）：
  前 `epn` 行是 home 种子、后 `epn` 行是副本 dW，行序与 `prefetch_weight` 视图
  一致；每组连续，组间数据类型一致。
- 梯度数据类型只能是 bf16 / fp16 / fp32，且必须等于构造期 `grad_dtype`
  定型值，运行期不允许漂移。
- 组数和各组行宽在首次调用后定型；`Wg` 须为 64 的倍数（尾块防护）。
- `slots` 为 None 时取构造定型值，显式给出时必须一致；`slot` 只做一致性
  校验，不影响累加落点。
- 时序：本算子要晚于本层权重最后一次读、早于下一层 `prefetch_weight`（单槽）。

## 3. 行为说明

- 发送侧由 host 把副本行（`local_grads` 后 `epn` 行）拷进本 rank
  workspace 的副本区 `[epn, Wg] × 组`；副本区大小恒定，不随 R 增长。
- owner 侧用一个 kernel 完成累加：头部屏障（把本 rank 的 epoch 写给全部
  peer，并等齐全部 peer 的信号）→ 各读块按 peer 轮转，从对端映射内存
  逐行远读 → UB 流水环 → 用拷贝引擎原子加直接累到本 rank 的 home 段
  （原位，不额外搬种子）：

  $$
  \mathbf{G}'_{home}[j,\ :] = \mathbf{G}_{seed}[j,\ :]
  + \sum_{r \in owners(j)} \mathbf{G}_r[j,\ :].
  $$

- 累加期不做 vf 运算：原子加按张量数据类型进行，bf16 / fp16 逐加舍入，
  fp32 为精确档。
- 头部屏障的写者和读块数都封顶在常驻核数内，并按 peer 轮转分配
  （R>64 时每块分管多个 peer）；写者全部落在首波，不依赖块派发顺序。
- kernel 入队后再追加一次组内屏障（入队形态，不做 host 同步），
  保证下一轮 host 侧副本行拷贝开始前，全组本轮累加都已结束；同步模式
  返回即本轮完成，`async_finish=True` 时由返回的发起流事件表达完成。
- 读块的行分片数与流水环槽深有缺省值（`MOONEP_ACC_SPLIT=8`、
  `MOONEP_ACC_DEPTH=3`），一般无需调整；改动会重新编译 kernel。

## 4. 用法

`MoonepBuffer` 路径：

```python
home_views = buf.reduce_grad(plan, local_grads=(dw13, dw2))
# 异步：home_views, event = buf.reduce_grad(
#     plan, local_grads=(dw13, dw2), async_finish=True,
# )
```

裸 API 路径（须按形状精算 workspace）：

```python
import torch
import torch.distributed as dist
from moonep_buffer import _workspace_need
from moonep_context import create_moonep_context
from reduce_grad import reduce_grad

R = dist.get_world_size()
epn = E // R
W = H * Hp
ctx = create_moonep_context(
    group=dist.group.WORLD,
    workspace_bytes=_workspace_need(R, epn, W, 1, torch.bfloat16, torch.bfloat16),
)
home_views = reduce_grad(ctx, copy_list, (dw13, dw2), epn)
```

`_workspace_need()` 为内部辅助函数（签名可能随版本调整）；最后两个参数是
梯度 / 权重 dtype，须与实际使用一致，否则容量会偏小。
