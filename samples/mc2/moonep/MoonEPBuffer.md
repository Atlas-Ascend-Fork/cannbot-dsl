# MoonEPBuffer

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

- 接口功能：`MoonepBuffer` 统一封装 `plan`、`prefetch_weight`、`reduce_grad`、`dispatch`、`combine` 五个算子。构造时一次完成 workspace 分配、UbMem 建连与数据类型/槽位定型；`dispatch` / `combine` 使用的 ElasticBuffer 在首次调用时懒构造。
- 规划与权重/梯度链路走 MoonEP 私有 workspace；token 数据面走 ElasticBuffer 的 UbMem 通路。
- `plan` / `prefetch_weight` / `reduce_grad` 返回路径不做 host 同步；视图与 `plan` 结果在 `destroy()` 前有效。

## 函数原型

```python
MoonepBuffer(
    S, H, Hp, K, E,
    *,
    num_ep_ranks=None,
    moonep_group,
    ep_group,
    weight_dtype="bf16",
    grad_dtype="bf16",
    slots=1,
)

plan(tokens_per_expert, topk_experts_sk) -> MoonepPlan
prefetch_weight(plan, async_finish=False, *, local_weights, local_scales=None, slot=0)
reduce_grad(plan, async_finish=False, *, local_grads)
dispatch(plan, x, topk_weights=None, *, do_cpu_sync=True)
combine(y, handle, *, topk_weights=None)
views(slot=0)
weight_view(group=0, slot=0)
barrier()
destroy(*, release_group=False, release_eb=False)
```

## 参数说明

<table>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>S</td>
      <td>输入</td>
      <td>每 rank token 数。</td>
      <td>≥ 1；各 rank 须一致；上限受显存/workspace 约束。</td>
      <td>INT</td>
    </tr>
    <tr>
      <td>H</td>
      <td>输入</td>
      <td>hidden size，同时是专家 FFN 输入维。</td>
      <td>≥ 1；上限受显存/workspace 约束。</td>
      <td>INT</td>
    </tr>
    <tr>
      <td>Hp</td>
      <td>输入</td>
      <td>专家 FFN 中间维，权重行宽 <code>W = H × Hp</code>。</td>
      <td>≥ 1；<code>W</code> 须为 64 的倍数；上限受显存/workspace 约束。</td>
      <td>INT</td>
    </tr>
    <tr>
      <td>K</td>
      <td>输入</td>
      <td>每 token top-k。</td>
      <td><code>1 <= K <= E</code>。</td>
      <td>INT</td>
    </tr>
    <tr>
      <td>E</td>
      <td>输入</td>
      <td>EP 组专家总数。</td>
      <td><code>1 <= E <= MAX_E</code>（1024）、<code>E % R == 0</code>。</td>
      <td>INT</td>
    </tr>
    <tr>
      <td>num_ep_ranks</td>
      <td>可选输入</td>
      <td>EP 组 rank 数。</td>
      <td>None 取组 world size；显式给出时须与组 world size 一致，且 2 ≤ num_ep_ranks ≤ 128。</td>
      <td>INT</td>
    </tr>
    <tr>
      <td>moonep_group</td>
      <td>输入</td>
      <td><code>plan</code> / <code>prefetch_weight</code> / <code>reduce_grad</code> 通信组。</td>
      <td>仅接受 ProcessGroup，不接受 str / None。</td>
      <td>ProcessGroup</td>
    </tr>
    <tr>
      <td>ep_group</td>
      <td>输入</td>
      <td><code>dispatch</code> / <code>combine</code>（ElasticBuffer）通信组。</td>
      <td>不能为 None；须与 <code>moonep_group</code> 分属两个独立通信器。</td>
      <td>ProcessGroup</td>
    </tr>
    <tr>
      <td>weight_dtype</td>
      <td>可选输入</td>
      <td>权重数据类型定型值。</td>
      <td><code>bf16</code> / <code>fp16</code> / <code>e4m3fn</code> / <code>e5m2</code>，缺省 <code>bf16</code>。</td>
      <td>STRING 或 torch.dtype</td>
    </tr>
    <tr>
      <td>grad_dtype</td>
      <td>可选输入</td>
      <td>梯度数据类型定型值。</td>
      <td><code>bf16</code> / <code>fp16</code> / <code>fp32</code>，缺省 <code>bf16</code>。</td>
      <td>STRING 或 torch.dtype</td>
    </tr>
    <tr>
      <td>slots</td>
      <td>可选输入</td>
      <td>权重窗口槽数。</td>
      <td>≥ 1；层间流水取 2。</td>
      <td>INT</td>
    </tr>
  </tbody>
</table>

其中 `R` 为组内 rank 数（`num_ep_ranks`），`W = H × Hp`。

## 接口说明

### plan

```python
plan(tokens_per_expert, topk_experts_sk) -> MoonepPlan
```

功能说明：规划入口，返回 `dst_n` / `cu_seqlens` / `experts_to_copy` / `src_n` / `vtopk` 五件套。

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| `tokens_per_expert` | 输入 | 本 rank 各专家 token 计数，shape `(E,)` | INT32 | ND |
| `topk_experts_sk` | 输入 | 本 rank 的 top-k 专家索引，shape `(S, K)`，取值 `[0, E)` | INT32 | ND |
| `dst_n` | 输出 | 零占位（vtopk-only 快路径不计算），shape `(N,)` | INT32 | ND |
| `cu_seqlens` | 输出 | 分组 GEMM 分段偏移，shape `(2*epn,)`，末元素等于本地 N | INT32 | ND |
| `experts_to_copy` | 输出 | 各 rank 的副本清单，shape `(R, epn)`，空槽为 -1，每行单源 | INT32 | ND |
| `src_n` | 输出 | 零占位（vtopk-only 快路径不计算），shape `(N,)` | INT32 | ND |
| `vtopk` | 输出 | 虚拟专家 id，shape `(N,)`，取值 `[0, 2E)` | INT32 | ND |

接口约束：

- 两入参必须为 int32；形状分别严格为 `[E]` 与 `[S, K]`；`tokens_per_expert` 与 `topk_experts_sk` 统计出的计数须一致。
- `E <= 1024`、`R <= 128`、`E % R == 0`；各 rank 的 S 须一致。
- 返回不做 host 同步；不均衡在下一个算子入口报错。

### prefetch_weight

```python
prefetch_weight(
    plan, async_finish=False, *,
    local_weights, local_scales=None, slot=0,
) -> (weights, scales) | ((weights, scales), event)
```

功能说明：把本 rank 权重推到各 rank 槽位。

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| `plan` | 输入 | 同一 `MoonepBuffer` 的 `plan()` 结果 | MoonepPlan | - |
| `async_finish` | 可选输入 | True 跳过末尾屏障的 host 同步，返回发起流事件 | BOOL | - |
| `local_weights` | 输入 | 2 或 3 组 home 权重，每组 `(epn, ...)`（至少 2 维，展平为 `(epn, Wg)`） | BFLOAT16、FLOAT16、FLOAT8_E4M3FN、FLOAT8_E5M2 | ND |
| `local_scales` | 可选输入 | mxfp8 的 scale 组，2 或 3 组，每组 `(epn, Sg)`（`Sg = ceil(Wg/32)`，32B 对齐），与权重同序等长 | FLOAT8_E8M0FNU | ND |
| `slot` | 可选输入 | 窗口槽位，缺省 0，须 `0 <= slot < slots` | INT | - |
| `weights` | 输出 | 各组 `[2*epn, Wg]` 视图（home 行 + 副本槽） | BFLOAT16、FLOAT16、FLOAT8_E4M3FN、FLOAT8_E5M2 | ND |
| `scales` | 输出 | 各组 scale 视图；未量化时为空元组 | FLOAT8_E8M0FNU | ND |
| `event` | 可选输出 | `async_finish=True` 时追加的发起流事件 | torch.npu.Event | - |

接口约束：

- 组数只能是 2 或 3，组间 dtype 一致且等于 `weight_dtype`；组数与各组行宽在首次调用后定型。
- mxfp8 双向硬契约：fp8 权重必传 `local_scales`；传了 `local_scales` 权重必须是 fp8；scale 行宽须为正且为 32B 倍数。
- 总宽不超过构造期按 `3W` 预留的容量；空槽填 `-1`（bf16/fp16/fp32），fp8 视图保留 0 初始化。
- 视图按 `slot` 缓存，可用 `views(slot)` / `weight_view(group, slot)` 取回。

### reduce_grad

```python
reduce_grad(
    plan, async_finish=False, *, local_grads,
) -> home_views | (home_views, event)
```

功能说明：把副本梯度累加回 home。

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| `plan` | 输入 | 同一 `MoonepBuffer` 的 `plan()` 结果（副本清单与 prefetch 共用） | MoonepPlan | - |
| `async_finish` | 可选输入 | True 返回发起流事件；False（缺省）阻塞到全组归并完成 | BOOL | - |
| `local_grads` | 输入 | 2 或 3 组 full-dW，每组 `(2*epn, Wg)` 或 `(2*epn, Hp, H)`：前 epn 行 home 种子、后 epn 行副本 dW | BFLOAT16、FLOAT16、FLOAT32 | ND |
| `home_views` | 输出 | 各组 `(epn, Wg)` 归并结果视图（与输入同存储）；dtype 同 `local_grads` | BFLOAT16、FLOAT16、FLOAT32 | ND |
| `event` | 可选输出 | `async_finish=True` 时追加的发起流事件 | torch.npu.Event | - |

接口约束：

- `local_grads` 必须是 2 或 3 组的 list/tuple；行序与 `prefetch_weight` 视图一致；组间 dtype 一致且等于构造期 `grad_dtype`。
- 组数与各组行宽在首次调用后定型；`Wg` 须为 64 的倍数。
- 时序：晚于本层权重最后一次读、早于下一层 `prefetch_weight`。

### dispatch

```python
dispatch(plan, x, topk_weights=None, *, do_cpu_sync=True) -> (recv_x, recv_w, handle)
```

功能说明：按 `plan.vtopk` 把 token 分发到各 rank。

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| `plan` | 输入 | `plan()` 返回的 `MoonepPlan`；内部取 `plan.vtopk` 作为路由 | MoonepPlan | - |
| `x` | 输入 | 前向 token 载荷 / 反向输出梯度，shape `(S, H)` | BFLOAT16 | ND |
| `topk_weights` | 可选输入 | 路由权重，shape `(S, K)`，内部转 fp32 | FLOAT32 | ND |
| `do_cpu_sync` | 可选输入 | True（缺省）精确分配接收区；False 按 worst-case 容量分配 | BOOL | - |
| `recv_x` | 输出 | 接收区，按专家分组，shape `(cap, H)`；dtype 同 `x` | BFLOAT16 | ND |
| `recv_w` | 输出 | 接收行对应的路由权重，shape `(cap,)`；`topk_weights=None` 时为 None | FLOAT32 | ND |
| `handle` | 输出 | 本次分发的路由结果，供对应 `combine` 使用 | EPHandle | - |

接口约束：

- `plan` 须为同一 `MoonepBuffer` 的 `plan()` 结果；`x` 形状严格为 `(S, H)`；`topk_weights` 形状严格为 `(S, K)`；张量须在 NPU 上。
- 内部用 `plan.vtopk.view(S, K)` 作为 `topk_idx`、`num_experts = 2E`、每 rank 容量 `nmtr = S`。
- 前向和反向的 handle 只与同方向 `combine` 配对。

### combine

```python
combine(y, handle, *, topk_weights=None) -> (out, out_w)
```

功能说明：把专家输出按路由回收。

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| `y` | 输入 | 专家输出，shape `(cap, H)`；`cap` 等于对应 `dispatch` 的接收容量；dtype 同 `recv_x` | BFLOAT16 | ND |
| `handle` | 输入 | 对应方向 `dispatch` 返回的 handle | EPHandle | - |
| `topk_weights` | 可选输入 | `None` = 裸和（反向）；传 `recv_w`（shape `(cap,)`）= 按权重回收（前向） | FLOAT32 | ND |
| `out` | 输出 | 回收结果，token 主序，shape `(S, H)`；dtype 同 `y` | BFLOAT16 | ND |
| `out_w` | 输出 | 回收的路由权重，shape `(S, K)`；`topk_weights=None` 时为 None | FLOAT32 | ND |

接口约束：

- `y` 须为二维、末维等于 `H`、在 NPU 上；有效行不足容量时须零填充到 `cap`。
- `handle` 必须来自同方向 `dispatch`；`topk_weights=None` 时 `out_w` 也为 None。

### views / weight_view

```python
views(slot=0) -> (weights, scales)
weight_view(group=0, slot=0) -> Tensor
```

功能说明：取 `prefetch_weight` 缓存的权重/scale 视图；`views` 返回整组二元组，`weight_view` 返回单组权重视图（GEMM 直读）。

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| `slot` | 可选输入 | 窗口槽位，缺省 0；须已成功调用过该槽位的 `prefetch_weight` | INT | 须满足 `0 <= slot < slots` |
| `group` | 可选输入 | 权重组下标，缺省 0 | INT | 须满足 `0 <= group < 组数`（2 或 3） |
| `weights` | 输出 | 各组 `[2*epn, Wg]` 视图（前 epn 行 home、后 epn 行副本） | BFLOAT16、FLOAT16、FLOAT8_E4M3FN、FLOAT8_E5M2（同 `weight_dtype`） | ND |
| `scales` | 输出 | 各组 `[2*epn, Sg]` scale 视图；未量化时为空元组 `()` | FLOAT8_E8M0FNU | ND |

接口约束：

- 组数语义与 `prefetch_weight` 入参一致：3 组等宽按 `(w1, w3, w2)` 对应 gate/up/down；2 组交织按 `(w13, w2)`，`weight_view(0)` 为 W13（每专家一行 2W，gate 前 up 后），`weight_view(1)` 为 down。
- 视图按 `slot` 缓存且同 `data_ptr` 幂等；未调用过 `prefetch_weight` 时 `views` 直接报错。
- 返回的视图与 `plan` 张量一样，须在 `destroy()` 前使用完毕。

### barrier

```python
barrier() -> None
```

功能说明：workspace flag 自旋 barrier；返回即全组此前流上的工作都已完成。

无入参，返回 `None`。

接口约束：

- 集体调用：全组 rank 须同序进入；调用前 `MoonepBuffer` 须处于存活状态（未 `destroy`）。

### destroy

```python
destroy(*, release_group=False, release_eb=False) -> None
```

功能说明：显式释放逻辑资源（幂等）；调用后本对象的方法经 `_require_alive` 拒绝，视图与 `plan` 张量失效。

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| `release_group` | 可选输入 | False（缺省）不销毁子组，留待进程退出回收；True 显式 `dist.destroy_process_group(子组)` | BOOL | True / False（缺省 False） |
| `release_eb` | 可选输入 | False（缺省）不销毁 EB，引用保留到进程退出；True 显式 `ElasticBuffer.destroy()`（约 120s 级耗时） | BOOL | True / False（缺省 False） |

接口约束：

- 幂等：重复调用直接返回；调用后 DSL 通信连接已断开（`connection.disconnect`），`workspace` 随张量引用回收。
- `release_group=True` 时调用方须保证全组 rank 同序执行；子组为注入对象，缺省不代销毁。
- `release_eb=True` 属显式确定性回收（慢路径），适合串行多 buffer 场景；缺省保留引用可避免 GC 中途触发 EB 析构造成分钟级停顿。
- `heap_bytes` / `heap_required_bytes` / `destroyed` 等只读属性在 destroy 后仍可读取。

## 约束说明

- **构造与通信域**：构造是集体调用，`moonep_group` / `ep_group` 的成员须以相同参数、相同顺序执行；两个组须为独立 HCCL 通信器。`dispatch` / `combine` 依赖 `cann_ops_transformer`（ElasticBuffer）；数据面不占通信域 HCCL 外部 buffer，`hccl_buffer_size` 由创建方自理。
- **环境**：构造前完成 `dist.init_process_group(backend="hccl")`；一进程一卡并绑定 `torch.npu.set_device(local_rank)`；单机 ≥ 2 卡，8 卡为验收形态。
- **规模与形状**：`E <= 1024`、`R <= 128`（R 为组内 rank 数，布局常量 `MAX_R`）、`E % R == 0`、`S >= 1`、`1 <= K <= E`；各 rank 的本地 token 数 S 须一致。
- **数据类型**：`tokens_per_expert` / `topk_experts_sk` / `experts_to_copy` 为 int32；权重支持 bf16 / fp16 / e4m3 / e5m2，组间一致且运行期不可漂移；梯度支持 bf16 / fp16 / fp32，组间一致且等于构造定型值。
- **权重形态**：2 组交织（gate/up 合并为 `W13`，行宽 `2W`；down 行宽 `W`）或 3 组等宽（`W, W, W`）；fp16 与 bf16 同为 2 字节、不配 scale；mxfp8 的 fp8 权重与 `float8_e8m0fnu` block scale 必须成对，scale 行宽 `ceil(Wg/32)` 字节且 32B 对齐。
- **槽位与容量**：`slots >= 1`，全链一致；权重/梯度总宽上限 `3W`，workspace 构造期按形状精算。
- **生命周期**：视图与 `plan` 返回的张量须在 `destroy()` 前用完；`destroy` 可重复调用。

## 调用示例

以下省略 `tpe` / `topk` / `x` / `scores` / `w13` / `w2` / `dw13` / `dw2` / `gmm_out_padded` 的构造（按上文形状与类型准备）。

```python
import os

import torch
import torch.distributed as dist
import torch_npu
from moonep_buffer import MoonepBuffer

dist.init_process_group(backend="hccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.npu.set_device(local_rank)

ranks = list(range(dist.get_world_size()))
moonep_group = dist.new_group(ranks=ranks, backend="hccl", use_local_synchronization=True)
ep_group = dist.new_group(ranks=ranks, backend="hccl", use_local_synchronization=True)

S, H, Hp, K, E = 4096, 7168, 2048, 8, 64
buf = MoonepBuffer(
    S, H, Hp, K, E,
    moonep_group=moonep_group,
    ep_group=ep_group,
    weight_dtype="bf16",
    grad_dtype="bf16",
    slots=2,
)

plan = buf.plan(tpe, topk)                    # tpe [E]、topk [S,K]，均为 int32
weights, scales = buf.prefetch_weight(plan, local_weights=(w13, w2))
recv_x, recv_w, handle = buf.dispatch(
    plan, x, topk_weights=scores, do_cpu_sync=False,
)
out, out_w = buf.combine(gmm_out_padded, handle, topk_weights=recv_w)
home_views = buf.reduce_grad(plan, local_grads=(dw13, dw2))

buf.destroy()
```
