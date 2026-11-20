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
    S, H, Hp, K, E, num_ep_ranks,
    num_sms=None, token_padding=128, group=None,
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
| `num_ep_ranks` | 输入 | EP 组 rank 数；None 取 `group` world size | int | ND |
| `num_sms` | 可选输入 | MoonEP kernel 常驻块数上限；None 按硬件常驻核数 | int | ND |
| `token_padding` | 可选输入 | 预留参数，当前不生效 | int | ND |
| `group` | 可选输入 | MoonEP 上下文与 ElasticBuffer 共用的通信组 | ProcessGroup | - |
| `weight_dtype` | 可选输入 | 权重数据类型定型值 | str 或 torch.dtype | ND |
| `grad_dtype` | 可选输入 | 梯度数据类型定型值，缺省 `bf16` | str 或 torch.dtype | ND |
| `slots` | 可选输入 | 权重窗口槽数，层间流水取 2 | int | ND |

构造是集体调用：内含 workspace 注册、UbMem 建连与 ElasticBuffer 创建；全 EP rank 须以相同参数、相同顺序执行。

接口约束：

- `S >= 1`、`H >= 1`、`Hp >= 1` 且 `W` 为 64 的倍数；`K >= 1`。
- `E <= 1024`、`E % R == 0`（R 为通信组 rank 数）；`num_ep_ranks` 显式给出时须等于组 world size。
- `group` 仅接受 `ProcessGroup`，None 取 `dist.group.WORLD`。
- `num_sms` 为正整数或 None，仅作用于 dispatch 内部的规划、prefetch_weight、reduce_grad。
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
| `hidden_sh` | 输入 | 前向 token 载荷 / 反向输出梯度，shape `(S, H)` | BFLOAT16 | ND |
| `route_weights_sk` | 可选输入 | 前向路由权重，shape `(S, K)`，内部转 fp32 | FLOAT32 | ND |
| `topk_experts_sk` | 可选输入 | 前向 top-k 专家索引，shape `(S, K)` | INT32 | ND |
| `tokens_per_expert` | 可选输入 | 前向专家计数，shape `(E,)` | INT32 | ND |
| `plan` | 可选输入 | 反向复用前向 plan；与路由参数互斥 | MoonepPlan | - |
| `do_cpu_sync` | 可选输入 | True（缺省）精确分配接收区；False 按 worst-case 容量分配 | BOOL | - |
| `hidden_nvsh` | 输出 | 接收区，shape `(cap, H)`；dtype 同输入 | BFLOAT16 | ND |
| `route_weights_nvs` | 输出 | 接收行路由权重，shape `(cap,)`；反向为 None | FLOAT32 | ND |
| `cu_seqlens` | 输出 | 前向 plan 的分段偏移（`plan.cu_seqlens`） | INT32 | ND |
| `plan` | 输出 | 前向新建 / 反向原样返回的 `MoonepPlan` | MoonepPlan | - |

接口约束：

- 前向须传 `topk_experts_sk` 与 `tokens_per_expert`，`route_weights_sk` 可选；反向必须传 `plan=` 且路由参数为空。
- `hidden_sh` 严格为 `(S, H)`；`route_weights_sk` / `topk_experts_sk` 严格为 `(S, K)`；张量须在 NPU 上。
- 内部用 `plan.vtopk.view(S, K)` 作为 `topk_idx`、`num_experts = 2E`、每 rank 容量 `nmtr = S`。

### prefetch_weight

```python
prefetch_weight(
    *, plan, local_gate_weight=None, local_up_weight=None,
    local_down_weight=None, local_gate_up_weight=None,
    local_gate_scales=None, local_up_scales=None, local_down_scales=None,
    local_gate_up_scales=None, async_finish=False, slot=0,
)

# 3 组
gate_pb, up_pb, down_pb = prefetch_weight(
    plan=plan,
    local_gate_weight=w1, local_up_weight=w3, local_down_weight=w2,
)
# W13 交织
gate_up_pb, down_pb = prefetch_weight(
    plan=plan, local_gate_up_weight=w13, local_down_weight=w2,
)
```

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| `plan` | 输入 | dispatch 返回的 `MoonepPlan` | MoonepPlan | - |
| `local_gate_weight` | 输入 | gate home 权重，shape `(epn, ...)` | 构造 `weight_dtype` | ND |
| `local_up_weight` | 输入 | up home 权重，shape `(epn, ...)` | 构造 `weight_dtype` | ND |
| `local_down_weight` | 输入 | down home 权重，shape `(epn, ...)`；W13 形态也必传 | 构造 `weight_dtype` | ND |
| `local_gate_up_weight` | 可选输入 | W13 交融权重，每专家一行 2W | 构造 `weight_dtype` | ND |
| `local_gate_scales` / `local_up_scales` / `local_down_scales` | 可选输入 | 3 组 mxfp8 scale，shape `(epn, Sg)` | FLOAT8_E8M0FNU | ND |
| `local_gate_up_scales` / `local_down_scales` | 可选输入 | W13 2 组 scale | FLOAT8_E8M0FNU | ND |
| `async_finish` | 可选输入 | True 返回发起流事件 | BOOL | - |
| `slot` | 可选输入 | 窗口槽位，`0 <= slot < slots` | int | - |
| `gate_pb` / `up_pb` / `down_pb` | 输出 | 3 组权重视图，各组 `[2*epn, Wg]` | 同权重 | ND |
| `gate_up_pb` / `down_pb` | 输出 | W13 形态二元组 | 同权重 | ND |

接口约束：

- 3 组（gate + up + down）与 W13（gate_up + down）二选一；W13 不接受 gate/up 独立参数。
- 组数与各组行宽在首次调用后定型，后续必须一致；组间 dtype 一致且等于构造 `weight_dtype`。
- mxfp8 双向契约：fp8 权重必须传 scales；传了 scales 权重就必须是 fp8；scale 类型固定 `float8_e8m0fnu`，行宽须为 32B 倍数。
- `0 <= slot < slots`；总宽不超过构造期按 `3W` 预留的容量；空槽填 `-1`（bf16/fp16/fp32），fp8 视图保留 0。
- scale 视图经 `scale_views(slot=0)` 读取；`async_finish=True` 时返回 `(buffers, event)`。

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
| `local_gate_grad` / `local_up_grad` / `local_down_grad` | 输入 | 3 组 full-dW，每组 `(2*epn, Wg)` 或 `(2*epn, Hp, H)` | 构造 `grad_dtype` | ND |
| `local_gate_up_grad` / `local_down_grad` | 输入 | W13 形态 2 组 full-dW | 构造 `grad_dtype` | ND |
| `async_finish` | 可选输入 | True 返回发起流事件 | BOOL | - |
| `gate_rb` / `up_rb` / `down_rb` | 输出 | 3 组归并结果，各组 `(epn, Wg)` | 同梯度 | ND |
| `gate_up_rb` / `down_rb` | 输出 | W13 形态二元组 | 同梯度 | ND |

接口约束：

- `local_grads` 为 2 或 3 组 full-dW：前 `epn` 行 home 种子、后 `epn` 行副本 dW，行序与 `prefetch_weight` 视图一致。
- 组间 dtype 一致且等于构造期 `grad_dtype`；`Wg` 须为 64 的倍数。
- 时序：晚于本层权重最后一次读、早于下一层 `prefetch_weight`；`async_finish=True` 时返回 `(buffers, event)`。

### 辅助接口

| 接口 | 说明 |
| :--- | :--- |
| `scale_views(slot=0)` | 最近一次 prefetch 的 scale 视图元组；未量化时为空元组 |
| `barrier()` | 组内屏障；返回即全组此前流上的工作都已完成 |
| `destroy(*, release_group=False, release_eb=False)` | 幂等释放；缺省不销毁组/EB，WORLD 组始终不销毁，传 `True` 才显式回收非 WORLD 组/EB |

## 约束说明

- **环境**：构造前完成 `dist.init_process_group(backend="hccl")`；一进程一卡并绑定 `torch.npu.set_device(local_rank)`；单机 ≥ 2 卡，8 卡为验收形态。
- **通信域**：`group` 为单个 `ProcessGroup`，同时用于 MoonEP 上下文与 ElasticBuffer；None 取 WORLD。数据面不占通信域 HCCL 外部 buffer，`hccl_buffer_size` 由创建方自理。
- **规模与形状**：`E <= 1024`、`R <= 128`、`E % R == 0`、`S >= 1`、`K >= 1`；各 rank 的 S 一致。
- **数据类型**：`tokens_per_expert` / `topk_experts_sk` / `experts_to_copy` 为 int32；权重支持 bf16 / fp16 / e4m3 / e5m2，组间一致；梯度支持 bf16 / fp16 / fp32 且等于构造定型值。
- **分组与行宽**：权重/梯度支持 3 组（常见等宽 `(W, W, W)`）或 W13 2 组（`(2W, W)`），总宽上限 `3W`（`W = H × Hp`）；`W` 须为 64 的倍数。
- **槽位**：`slots >= 1`，全链一致；`token_padding` 当前仅保留参数，不生效。
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
gate_pb, up_pb, down_pb = buf.prefetch_weight(
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

完整的权重形态（3 组 / W13 / fp16 / FP8）见下文。

### 裸 API 用法

```python
from moonep_buffer import _workspace_need
from moonep_context import create_moonep_context
from moonep_plan import moonep_plan
from prefetch_weight import prefetch_weight
from reduce_grad import reduce_grad

ctx = create_moonep_context(
    group=dist.group.WORLD,
    workspace_bytes=_workspace_need(
        dist.get_world_size(), E // dist.get_world_size(), W, 1,
        torch.bfloat16, torch.bfloat16,
    ),
)
topk = topk_idx.to(torch.int32).npu()
tpe = torch.bincount(topk.cpu().flatten(), minlength=E).to(torch.int32).npu()
dst_n, cu_seqlens, copy_list, src_n, vtopk = moonep_plan(ctx, tpe, topk)
weights, scales = prefetch_weight(ctx, copy_list, (w13, w2), epn)
home_views = reduce_grad(ctx, copy_list, (dw13, dw2), epn)
```

`create_moonep_context` 缺省只分配 plan 交换区下界；`prefetch_weight` / `reduce_grad` 的形状相关区域由调用方按 `_workspace_need()` 传入。

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
