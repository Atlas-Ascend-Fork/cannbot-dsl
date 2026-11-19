# prefetch_weight

`prefetch_weight` 按 `moonep_plan` 产出的副本清单，把本 rank 的 home 权重
直推到各 rank 的预取槽，返回 `[2*epn, Wg]` 的“home 行 + 副本槽”视图。
`MoonepBuffer.prefetch_weight()` 透传本算子并缓存视图（配合
`views()` / `weight_view()`）。

## 1. 接口

```python
prefetch_weight(
    context, experts_to_copy, local_weights, epn,
    *, local_scales=None, slot=0, slots=1, async_finish=False,
)
```

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| `context` | 输入 | MoonEP 通信上下文 | - | - |
| `experts_to_copy` | 输入 | `moonep_plan` 输出的副本清单，shape `(R, epn)`（连续、NPU） | INT32 | ND |
| `local_weights` | 输入 | 本 rank home 权重，2 或 3 组，每组 shape `(epn, ...)`（至少 2 维，展平为 `(epn, Wg)`） | BFLOAT16、FLOAT16、FLOAT8_E4M3FN、FLOAT8_E5M2 | ND |
| `epn` | 输入 | 每 rank 本地专家数 `E / R` | INT | - |
| `local_scales` | 可选输入 | mxfp8 的 scale 组，2 或 3 组，每组 shape `(epn, Sg)`（`Sg = ceil(Wg/32)`，32B 对齐），与权重同序等长 | FLOAT8_E8M0FNU | ND |
| `slot` | 可选输入 | 窗口槽位，缺省 0 | INT | - |
| `slots` | 可选输入 | 窗口槽数，缺省 1；层间流水取 2 | INT | - |
| `async_finish` | 可选输入 | 异步完成开关，缺省 False | BOOL | - |
| `weights` | 输出 | 权重视图元组，各组 shape `(2*epn, Wg)`（home 行 + 副本槽） | 同 `local_weights` | ND |
| `scales` | 输出 | scale 视图元组，各组 shape `(2*epn, Sg)`；未量化时为空元组 | FLOAT8_E8M0FNU | ND |
| `event` | 可选输出 | `async_finish=True` 时追加的发起流事件 | - | - |

## 2. 接口约束

- 权重只能是 2 组或 3 组（kernel 入口按组数固定）；每组至少 2 维、首维
  严格等于 `epn`、连续；组间数据类型必须一致，且落在白名单内。
- 组数与各组行宽在首次调用后定型，后续调用必须一致（否则发布区/槽区会
  重叠，直接报错）。
- mxfp8 双向契约：fp8 权重必须传 `local_scales`；传了 `local_scales`
  权重就必须是 fp8。scale 数据类型固定 `float8_e8m0fnu`，语义宽度为
  `ceil(Wg / 32)` 字节（每 32 个权重 1 个 ue8m0）；实际行宽必须为正、
  且是 32B 的倍数（不足时调用方 pad 到 32B 倍数，消费端按实际行宽解）。
- `0 <= slot < slots`；`slots` 与构造定型值一致。
- 总宽不超过构造期按 `3W` 预留的容量，超出时容量校验报错。
- 权重行宽 `Wg` 的建议口径是构造期 `W = H × Hp`；2 组交织时第一组行宽为
  `2W`，按原始组宽校验容量。

## 3. 行为说明

- 输出视图恒为 `[2*epn, Wg]`：前 `epn` 行是 home 行（与本 rank 权重同值），
  后 `epn` 行是副本槽，非空槽由推送写入；空槽在视图创建时填 `-1`
  （bf16/fp16/fp32），fp8 视图保留 0 初始化。
- 首调传权重本体：创建视图并把参数拷进 home 行；之后传回
  `weights[g][:epn]`（与视图同址）则跳过拷贝，传独立张量则每次同步拷一次。
- 推送按对端 rank 组织：每块负责若干 peer，先扫 `experts_to_copy` 对应行
  得到源行号（-1 为空槽），再逐行把权重组从源地址经 UB 中转写到对端槽。
  常驻核数足够时每 peer 由最多 8 个 block 做行分片并发驱动（`peer = blk %
  n_rem`、`slice = blk // n_rem`），各分片在本端置完成标志，slice-0 汇总
  后才向对端写原有的 per-sender 信号，all-to-all 屏障语义不变。
- 每个 peer 的数据写完后，用 MTE3 同通道写一个本轮 epoch 信号；信号与
  数据同通道保序，对端收到信号即数据已全部落地。随后等待对端回写同轮
  信号，收尾同步就在 kernel 内完成。
- 同一进程内按 `R / epn / 各组宽度 / 各组元素字节宽 / 组数 / 窗口字节数 /
  常驻块数 / 每块负责的 peer 数` 缓存编译结果；本 rank 号不进编译键，运行时
  从上下文读取。
- `async_finish=True` 时返回发起流事件，`wait_event` 汇合后可复用缓冲区，
  便于层间流水（配 `slots=2`）。
- 分块推送统一走运行时循环（32KB 一块），编译产物尺寸不随行宽增长；
  大行（如 `W = 7168 × 3072`）不会再出现 IR 规模爆炸。

## 4. 用法

`MoonepBuffer` 路径：

```python
weights, scales = buf.prefetch_weight(plan, local_weights=(w13, w2))
# fp8：weights, scales = buf.prefetch_weight(
#     plan, local_weights=(w13, w2), local_scales=(s13, s2),
# )
weights_now = buf.views()[0]        # 缓存访问器（同一次调用的返回）
w2_view = buf.weight_view(group=1)  # 单组 [2*epn, Wg]
```

裸 API 路径（须按形状精算 workspace）：

```python
import torch
import torch.distributed as dist
from moonep_buffer import _workspace_need
from moonep_context import create_moonep_context
from prefetch_weight import prefetch_weight

R = dist.get_world_size()
epn = E // R
W = H * Hp
ctx = create_moonep_context(
    group=dist.group.WORLD,
    workspace_bytes=_workspace_need(R, epn, W, 1, torch.bfloat16, torch.bfloat16),
)
weights, scales = prefetch_weight(ctx, copy_list, (w13, w2), epn)
```

`_workspace_need()` 为内部辅助函数（签名可能随版本调整）；最后两个参数是
梯度 / 权重 dtype，须与实际使用一致，否则容量会偏小。
