# MoonEP

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950DT系列产品</term> | √ |
| <term>Ascend 950PR系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

MoonEP 是面向 EP（Expert Parallelism）的通信库：用动态冗余专家让各 rank 的 token 负载完全均衡。

- 处理链路：`dispatch` 内部先做规划（生成 `cu_seqlens`、`experts_to_copy`、`vtopk` 等）再分发 token；`prefetch_weight` 把本 rank 权重推到各 rank 槽位，`combine` 回收 token 输出，`reduce_grad` 把副本梯度累加回 home。
- 规划结果可跨调用、跨前反向复用；`plan` / `prefetch_weight` / `reduce_grad` 返回路径不做 host 同步。
- 全部计算为整型；固定输入下输出逐位一致（确定性）。
- `dispatch` / `combine` 由 `cann_ops_transformer` 的 ElasticBuffer 承载，其余三个算子走 MoonEP 私有 workspace。

`Buffer` 封装 `dispatch`、`prefetch_weight`、`combine`、`reduce_grad` 四个对外接口；规划（plan）在 `dispatch` 内部完成，不单独对外。同一个 `group` 同时用于 MoonEP 上下文与 ElasticBuffer；反向 `dispatch` 直接传 `plan=` 复用规划结果。构造、参数与调用示例见下文。

## 参数说明

### Buffer 构造

```python
Buffer(
    S, H, Hp, K, E, num_ep_ranks, group,
    num_sms=None, token_padding=128,
    *, weight_dtype="bf16", grad_dtype="bf16", slots=1,
)
```

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| `S` | 输入 | 每 rank token 数 | int | ND |
| `H` | 输入 | hidden size，同时是专家 FFN 输入维 | int | ND |
| `Hp` | 输入 | 专家 FFN 中间维，权重行宽 `W = H × Hp` | int | ND |
| `K` | 输入 | 每 token top-k | int | ND |
| `E` | 输入 | EP 组专家总数 | int | ND |
| `num_ep_ranks` | 输入 | 必选；EP 组 rank 数，取值 2–128，须等于 `group` 的 world size | int | ND |
| `group` | 输入 | 必选；MoonEP 上下文与 ElasticBuffer 共用的通信组 | ProcessGroup | - |
| `num_sms` | 可选输入 | 预留参数：MoonEP kernel 使用的 AIV 核数，取值 1–设备 AIV 核数；当前仅支持 `None` | int | ND |
| `token_padding` | 可选输入 | 预留参数，当前不生效 | int | ND |
| `weight_dtype` | 可选输入 | 权重数据类型定型值 | str 或 torch.dtype | ND |
| `grad_dtype` | 可选输入 | 梯度数据类型定型值，缺省 `bf16` | str 或 torch.dtype | ND |
| `slots` | 可选输入 | 权重窗口槽数，层间流水取 2 | int | ND |

构造是集体调用：内含 workspace 注册与 UbMem 建连；ElasticBuffer 在首次 `dispatch` / `combine` 时懒构造。全 EP rank 须以相同参数、相同顺序执行。

接口约束：

- `S >= 1`、`H >= 1`、`Hp >= 1` 且 `W` 为 64 的倍数；`K >= 1`。
- `num_ep_ranks` 取值 2–128，且须等于 `group` 的 world size；`E <= 1024`、`E % R == 0`（R 即 `num_ep_ranks`）。
- `group` 为必选参数，仅接受 `ProcessGroup`。
- `num_sms` 为预留参数，当前仅支持 `None`（按设备全部 AIV 核运行）；传其他值会被拒绝。
- `token_padding` 为预留参数，当前不生效。
- `weight_dtype` 取 `bf16` / `fp16` / `e4m3fn` / `e5m2`；`grad_dtype` 取 `bf16` / `fp16` / `fp32`。
- `slots >= 1`，全链一致。

### dispatch

```python
dispatch(hidden_sh, route_weights_sk=None, topk_experts_sk=None,
         tokens_per_expert=None, *, plan=None, do_cpu_sync=True)

# 前向：建 plan + 分发
hidden_nvsh, route_weights_nvs, cu_seqlens, plan = dispatch(
    hidden_sh, route_weights_sk, topk_experts_sk, tokens_per_expert,
)
# 反向：复用前向 plan
grad_expert_output_nvsh, _, _, _ = dispatch(grad_output_sh, plan=plan)
```

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| `hidden_sh` | 输入 | 前向 token / 反向输出梯度，shape `(S, H)` | BFLOAT16、FLOAT16 | ND |
| `route_weights_sk` | 可选输入 | 前向路由权重，shape `(S, K)`，内部转 fp32 | FLOAT32 | ND |
| `topk_experts_sk` | 可选输入 | 前向 top-k 专家索引，shape `(S, K)` | INT32 | ND |
| `tokens_per_expert` | 可选输入 | 前向专家计数，shape `(E,)` | INT32 | ND |
| `plan` | 可选输入 | 反向复用前向 plan；与路由参数互斥 | MoonepPlan | - |
| `do_cpu_sync` | 可选输入 | True（缺省）精确分配接收区；False 按 worst-case 容量分配 | BOOL | - |
| `hidden_nvsh` | 输出 | 接收区，shape `(cap, H)`；dtype 同输入 | 同输入 | ND |
| `route_weights_nvs` | 输出 | 接收行路由权重，shape `(cap,)`；反向为 None | FLOAT32 | ND |
| `cu_seqlens` | 输出 | 前向 plan 的分段偏移（`plan.cu_seqlens`） | INT32 | ND |
| `plan` | 输出 | 前向新建 / 反向原样返回的 `MoonepPlan` | MoonepPlan | - |

接口约束：

- 前向须传 `topk_experts_sk` 与 `tokens_per_expert`，`route_weights_sk` 可选；反向必须传 `plan=` 且路由参数为空。
- `hidden_sh` 严格为 `(S, H)`；`route_weights_sk` / `topk_experts_sk` 严格为 `(S, K)`；张量须在 NPU 上。
- `hidden_sh` 须为 bf16/fp16 且连续（与 `combine` 的 `hidden_nvsh` 同规则）。
- 内部用 `plan.vtopk.view(S, K)` 作为 `topk_idx`、`num_experts = 2E`、每 rank 容量 `nmtr = S`。

### MoonepPlan 结构

`MoonepPlan` 是 `dispatch` 前向返回的规划句柄，实现为 `NamedTuple`：字段顺序固定，可按 5 元组解包，也可按属性访问。

| 字段 | 形状 | 数据类型 | 说明 |
| :--- | :--- | :--- | :--- |
| `dst_n` | `(N,)` | INT32 | 零占位；vtopk-only 快路径不计算，恒为 0 |
| `cu_seqlens` | `(2*epn,)` | INT32 | 各 rank 的分段偏移，GMM 按此切分专家段 |
| `experts_to_copy` | `(R, epn)` | INT32 | 副本清单：第 `r` 行第 `s` 列是 rank `r` 第 `s` 个专家槽位的源 rank；`-1` 表示该槽位没有远程副本 |
| `src_n` | `(N,)` | INT32 | 零占位，同 `dst_n` |
| `vtopk` | `(N,)` | INT32 | 虚拟专家 id（扩展空间 `2E`），dispatch / combine 的路由依据 |

其中 `N = S × K`、`epn = E / R`、`R = num_ep_ranks`。

- `experts_to_copy` 同时是 `prefetch_weight` 与 `reduce_grad` 的输入；`vtopk.view(S, K)` 作为 `dispatch` 的 `topk_idx`。
- 返回路径不做 host 同步：溢出与布局不变量延迟到下一个 MoonEP 算子尾部校验。
- `plan` 可跨调用、跨前向/反向复用；同方向 `dispatch → combine` 之间不能再插一次同方向 `dispatch`。

### prefetch_weight

```python
prefetch_weight(
    *, plan, local_gate_weight=None, local_up_weight=None,
    local_down_weight=None, local_gate_up_weight=None,
    local_gate_scales=None, local_up_scales=None, local_down_scales=None,
    local_gate_up_scales=None, async_finish=False, slot=0,
)

# 3 组：返回值依次是权重视图元组与 scale 视图元组
(gate_pb, up_pb, down_pb), (gate_sb, up_sb, down_sb) = prefetch_weight(
    plan=plan,
    local_gate_weight=w1, local_up_weight=w3, local_down_weight=w2,
    local_gate_scales=s1, local_up_scales=s3, local_down_scales=s2,
)
# W13 交织
(gate_up_pb, down_pb), (gate_up_sb, down_sb) = prefetch_weight(
    plan=plan, local_gate_up_weight=w13, local_down_weight=w2,
    local_gate_up_scales=s13, local_down_scales=s2,
)
```

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| `plan` | 输入 | dispatch 返回的 `MoonepPlan` | MoonepPlan | - |
| `local_gate_weight` | 可选输入 | gate home 权重，shape `(epn, ...)`；3 组形态必传 | 构造 `weight_dtype` | ND |
| `local_up_weight` | 可选输入 | up home 权重，shape `(epn, ...)`；3 组形态必传 | 构造 `weight_dtype` | ND |
| `local_down_weight` | 输入 | down home 权重，shape `(epn, ...)`；3 组与 W13 形态都必传 | 构造 `weight_dtype` | ND |
| `local_gate_up_weight` | 可选输入 | W13 交织权重，每专家一行 2W；与 gate/up 二选一 | 构造 `weight_dtype` | ND |
| `local_gate_scales` | 可选输入 | gate 的 mxfp8 scale，shape `(epn, Sg)`；3 组形态使用 | FLOAT8_E8M0FNU | ND |
| `local_up_scales` | 可选输入 | up 的 mxfp8 scale，shape `(epn, Sg)`；3 组形态使用 | FLOAT8_E8M0FNU | ND |
| `local_down_scales` | 可选输入 | down 的 mxfp8 scale，shape `(epn, Sg)`；3 组与 W13 形态都使用 | FLOAT8_E8M0FNU | ND |
| `local_gate_up_scales` | 可选输入 | W13 交织 scale，行宽覆盖 2W；仅 W13 形态使用 | FLOAT8_E8M0FNU | ND |
| `async_finish` | 可选输入 | True 返回发起流事件 | BOOL | - |
| `slot` | 可选输入 | 窗口槽位，`0 <= slot < slots` | int | - |
| `gate_pb` | 输出 | gate 权重视图 `[2*epn, Wg]`；3 组形态返回 | 同权重 | ND |
| `up_pb` | 输出 | up 权重视图 `[2*epn, Wg]`；3 组形态返回 | 同权重 | ND |
| `down_pb` | 输出 | down 权重视图 `[2*epn, Wg]`；3 组与 W13 形态都返回 | 同权重 | ND |
| `gate_up_pb` | 输出 | W13 交织权重视图，行宽为 2W；仅 W13 形态返回 | 同权重 | ND |
| `scales` | 输出 | scale 视图元组，与权重逐组对应，行宽 `Sg`；未量化时为空元组 | FLOAT8_E8M0FNU | ND |

接口约束：

- 3 组（gate + up + down）与 W13（gate_up + down）二选一；W13 不接受 gate/up 独立参数。
- 组数与各组行宽在首次调用后定型，后续必须一致；组间 dtype 一致且等于构造 `weight_dtype`。
- mxfp8 双向契约：fp8 权重必须传 scales；传了 scales 权重就必须是 fp8；scale 类型固定 `float8_e8m0fnu`，行宽须为 32B 倍数。
- `0 <= slot < slots`；总宽不超过构造期按 `3W` 预留的容量；**空槽内容未定义**（等价 `torch.empty`，可能为零或上一轮残留值）——空槽在 `cu_seqlens` 中段长恒为 0，不参与计算。
- 返回 `(views, scales)` 二元组：`views` 为权重视图元组（3 组或 2 组），`scales` 为逐组对应的 scale 视图元组，未量化时为空元组；`async_finish=True` 时返回 `((views, scales), event)`。

### combine

```python
combine(*, plan, hidden_nvsh, route_weights_nvs=None)

output_sh, gathered_route_weights_sk, _ = combine(
    plan=plan, hidden_nvsh=expert_output_nvsh,
    route_weights_nvs=route_weights_nvs,
)
grad_hidden_sh, _, _ = combine(plan=plan, hidden_nvsh=grad_hidden_nvsh)
```

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| `plan` | 输入 | 同方向 dispatch 使用的 `MoonepPlan` | MoonepPlan | - |
| `hidden_nvsh` | 输入 | 专家输出，shape `(cap, H)`；尾行须零填充到 cap | BFLOAT16、FLOAT16 | ND |
| `route_weights_nvs` | 可选输入 | 前向传 dispatch 的权重；反向不传 | FLOAT32 | ND |
| `output_sh` | 输出 | token 主序回收结果，shape `(S, H)` | 同输入 | ND |
| `gathered_route_weights_sk` | 输出 | 回收权重，shape `(S, K)`；反向为 None | FLOAT32 | ND |
| `_` | 输出 | 固定 None（不暴露内部 handle） | - | - |

接口约束：

- 无 `handle` 入参：带 `route_weights_nvs` 为前向；不传时按未消费的 dispatch handle 推断方向；若前向/反向 handle 同时存在则报歧义错误。同方向 `dispatch → combine` 之间不能再插同方向 dispatch。
- `hidden_nvsh` 须为 bf16/fp16 且连续；`route_weights_nvs` 须为连续 1D float32，长度等于 `hidden_nvsh` 行数。
- 前向与反向的 handle 不能混用；`route_weights_nvs=None` 时 `gathered_route_weights_sk` 也为 None。

### reduce_grad

```python
reduce_grad(
    *, plan, local_gate_grad=None, local_up_grad=None,
    local_down_grad=None, local_gate_up_grad=None, async_finish=False,
)

# 3 组
gate_rb, up_rb, down_rb = reduce_grad(
    plan=plan,
    local_gate_grad=dw1, local_up_grad=dw3, local_down_grad=dw2,
)
# W13 交织
gate_up_rb, down_rb = reduce_grad(
    plan=plan, local_gate_up_grad=dw13, local_down_grad=dw2,
)
```

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| `plan` | 输入 | dispatch 返回的 `MoonepPlan` | MoonepPlan | - |
| `local_gate_grad` | 可选输入 | gate 的 full-dW，`(2*epn, Wg)` 或 `(2*epn, Hp, H)`；3 组形态必传 | 构造 `grad_dtype` | ND |
| `local_up_grad` | 可选输入 | up 的 full-dW，`(2*epn, Wg)` 或 `(2*epn, Hp, H)`；3 组形态必传 | 构造 `grad_dtype` | ND |
| `local_down_grad` | 输入 | down 的 full-dW，`(2*epn, Wg)` 或 `(2*epn, Hp, H)`；3 组与 W13 形态都必传 | 构造 `grad_dtype` | ND |
| `local_gate_up_grad` | 可选输入 | W13 交织 full-dW，行宽为 2W；仅 W13 形态必传 | 构造 `grad_dtype` | ND |
| `async_finish` | 可选输入 | True 返回发起流事件 | BOOL | - |
| `gate_rb` | 输出 | gate 归并结果 `(epn, Wg)`；3 组形态返回 | 同梯度 | ND |
| `up_rb` | 输出 | up 归并结果 `(epn, Wg)`；3 组形态返回 | 同梯度 | ND |
| `down_rb` | 输出 | down 归并结果 `(epn, Wg)`；3 组与 W13 形态都返回 | 同梯度 | ND |
| `gate_up_rb` | 输出 | W13 交织归并结果，行宽为 2W；仅 W13 形态返回 | 同梯度 | ND |

接口约束：

- 梯度为 2 或 3 组 full-dW：前 `epn` 行 home 种子、后 `epn` 行副本 dW，行序与 `prefetch_weight` 视图一致。
- 组间 dtype 一致且等于构造期 `grad_dtype`；`Wg` 须为 64 的倍数。
- 精度界：副本梯度在片上以 fp32 累加（16 位梯度归并期转 fp32）后单次写回原 dtype——bf16（尾数 7 位）误差 <= 2^-8；fp16（尾数 10 位）误差 <= 2^-11；fp32（尾数 23 位）误差 <= (R-1)·2^-24，即 R-1 次 fp32 加法的舍入累积。固定输入下逐元素加法序列恒定，输出逐位可重复。
- 时序：晚于本层权重最后一次读、早于下一层 `prefetch_weight`；`async_finish=True` 时返回 `(buffers, event)`。

### 辅助接口

- `barrier()`：组内对称堆自旋屏障（epoch 单调递增，flag 与数据同通道保序，不经 HCCL 集合通信）。返回时全组各 rank 在此前流上发起的工作都已完成。典型用法：① 阶段切换或调试时确认全组就绪；② 复用对端可读写的 MoonEP 缓冲（对称堆窗口、`prefetch_weight` 副本槽）前做跨 rank 先后保证——如多轮权重预取中"本 rank 写副本槽 → `barrier()` → 下一轮推送"，以及 `async_finish=True` 路径在复用或校验窗口前先对齐全组。
- `destroy(*, release_group=False, release_eb=False)`：幂等释放。缺省不销毁通信组与 ElasticBuffer，WORLD 组始终不销毁；传 `True` 才显式回收非 WORLD 组 / EB。
- 查询属性：`rank`、`rank_size`、`epn`、`H`、`Hp`、`slots`、`weight_dtype`、`grad_dtype`、`layout_groups`、`heap_required_bytes`、`heap_bytes`、`destroyed`，以及预留参数当前值 `num_sms`、`token_padding`。

## 约束说明

- **环境**：构造前完成 `dist.init_process_group(backend="hccl")`；一进程一卡并绑定 `torch.npu.set_device(local_rank)`；单机 ≥ 2 卡。
- **通信域**：`group` 为必选参数，单个 `ProcessGroup`，同时用于 MoonEP 上下文与 ElasticBuffer。数据面不占通信域 HCCL 外部 buffer，`hccl_buffer_size` 由创建方自理。
- **规模与形状**：`2 <= num_ep_ranks <= 128` 且等于通信组 world size；`E <= 1024`、`E % R == 0`、`S >= 1`、`K >= 1`；各 rank 的 S 一致。
- **数据类型**：`tokens_per_expert` / `topk_experts_sk` / `experts_to_copy` 为 int32；权重支持 bf16 / fp16 / e4m3 / e5m2，组间一致；梯度支持 bf16 / fp16 / fp32 且等于构造定型值；`dispatch` / `combine` 的 token 支持 bf16 / fp16。
- **分组与行宽**：权重/梯度支持 3 组（常见等宽 `(W, W, W)`）或 W13 2 组（`(2W, W)`），总宽上限 `3W`（`W = H × Hp`）；`W` 须为 64 的倍数。
- **槽位**：`slots >= 1`，全链一致；`num_sms`、`token_padding` 为预留参数，当前分别只接受 `None` 与缺省值。
- **生命周期**：视图和 `plan` 返回的张量须在 `destroy()` 前用完；`destroy` 可重复调用。

## 调用说明

### 通信域创建

```python
import torch.distributed as dist

ranks = list(range(dist.get_world_size()))
group = dist.new_group(
    ranks=ranks, backend="hccl", use_local_synchronization=True,
)
```

### 快速开始

```python
import torch
import torch.distributed as dist
import torch_npu
from moonep_buffer import Buffer

S, H, Hp, K, E = 4096, 7168, 2048, 8, 64
buf = Buffer(
    S, H, Hp, K, E, num_ep_ranks=dist.get_world_size(),
    group=group, slots=2,
)

hidden_nvsh, route_weights_nvs, cu_seqlens, plan = buf.dispatch(
    hidden_sh, route_weights_sk, topk_experts_sk, tokens_per_expert,
)
# 未量化时第二个返回值是空元组
(gate_pb, up_pb, down_pb), _ = buf.prefetch_weight(
    plan=plan,
    local_gate_weight=w1, local_up_weight=w3, local_down_weight=w2,
)
output_sh, gathered_route_weights_sk, _ = buf.combine(
    plan=plan, hidden_nvsh=expert_output_nvsh,
    route_weights_nvs=route_weights_nvs,
)
grad_expert_output_nvsh, _, _, _ = buf.dispatch(grad_output_sh, plan=plan)
grad_hidden_sh, _, _ = buf.combine(plan=plan, hidden_nvsh=grad_hidden_nvsh)
gate_rb, up_rb, down_rb = buf.reduce_grad(
    plan=plan,
    local_gate_grad=dw1, local_up_grad=dw3, local_down_grad=dw2,
)
buf.destroy()
```

### 测试与性能

每个测试文件须独立进程 / 作业运行。

```bash
# 规划（E2E golden + count audit）
python -m torch.distributed.run --nproc_per_node=8 -m pytest test/mc2/moonep/test_moonep_plan.py -q

# prefetch_weight / reduce_grad / Buffer（4 卡）
python -m torch.distributed.run --nproc_per_node=4 -m pytest test/mc2/moonep/test_prefetch_weight.py -q
python -m torch.distributed.run --nproc_per_node=4 -m pytest test/mc2/moonep/test_reduce_grad.py -q
python -m torch.distributed.run --nproc_per_node=4 -m pytest test/mc2/moonep/test_moonep_buffer.py -q

# 端到端与 token 数据面（EB）
python -m torch.distributed.run --nproc_per_node=4 -m pytest test/mc2/moonep/test_e2e.py -q
python -m torch.distributed.run --nproc_per_node=8 -m pytest test/mc2/moonep/test_moonep_ep.py -q

# R>64 布局 / 块划分静态回归（不占卡）
python -m pytest test/mc2/moonep/test_moonep_large_r.py -q
```
