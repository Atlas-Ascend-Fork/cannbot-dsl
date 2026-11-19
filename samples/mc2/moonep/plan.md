# moonep_plan（plan）

`moonep_plan` 是 MoonEP 的规划入口：跨 rank 交换专家需求，按每 rank 恰好
承载本地 `N = S*K` 行做贪心均衡，产出后续 `prefetch_weight` /
`reduce_grad` / `dispatch` 共用的副本清单与路由。

`MoonepBuffer.plan()` 透传本算子，并额外做构造期形状守卫（`tpe` 必须为
`[E]`、`topk` 必须为 `[S, K]`）。

## 1. 接口

```python
moonep_plan(context, tokens_per_expert, topk_experts_sk)
```

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| `context` | 输入 | MoonEP 通信上下文 | - | - |
| `tokens_per_expert` | 输入 | 本 rank 各专家 token 计数，shape `(E,)` | INT32 | ND |
| `topk_experts_sk` | 输入 | 本 rank 的 top-k 专家索引，shape `(S, K)`，取值 `[0, E)` | INT32 | ND |
| `dst_n` | 输出 | 零占位（vtopk-only 快路径不计算），shape `(N,)` | INT32 | ND |
| `cu_seqlens` | 输出 | 分组 GEMM 分段偏移，shape `(2*epn,)`，末元素等于本地 N | INT32 | ND |
| `experts_to_copy` | 输出 | 各 rank 的副本清单，shape `(R, epn)`，空槽为 -1，每行单源 | INT32 | ND |
| `src_n` | 输出 | 零占位（vtopk-only 快路径不计算），shape `(N,)` | INT32 | ND |
| `vtopk` | 输出 | 虚拟专家 id，shape `(N,)`，取值 `[0, 2E)` | INT32 | ND |

## 2. 接口约束

- `tokens_per_expert` 与 `topk_experts_sk` 都必须是 int32（`bincount` /
  `topk` 默认 int64，需显式转换）；形状分别严格为 `[E]` 与 `[S, K]`。
- `E <= 1024`、`R <= 128`、`E % R == 0`；`topk_experts_sk` 的取值须在
  `[0, E)`，且 `tokens_per_expert` 与它统计出的计数一致。
- 各 rank 的 S 必须一致；不均衡会在下一次算子的
  `check_pending_overflow` 处报错。
- 输入张量须在 NPU 上且连续；`experts_to_copy` 形状 `[R, epn]`，空槽为 -1。

## 3. 行为说明

- 先跨 rank 交换各专家需求，再按 `cap = N`（每 rank 恰好承载本地行数）做
  贪心均衡：超载 rank 把尚未服务的专家 token 段迁给最空闲 rank，直到全部
  rank 达到 `cap`。
- 每个目的 rank 至多接收一个源 rank 的副本（一卡一源），因此
  `experts_to_copy` 每行都是单源的；空槽写 -1。
- 分段与虚拟 id：`cu[j]` 是前 j 个槽位的 token 累计，`vtopk[i]` =
  服务 token i 的 rank × `2*epn` + 槽位号，落在 `[0, 2E)`。
- 返回时不与 host 同步；`experts_to_copy` / `vtopk` 等输出可跨调用、跨前反向复用。

实现备注：`epn == 1`（即 `E == R`）时，负载统计的内层循环恰为单次迭代，
canonicalize 折叠后事件会移出内层循环，触发
`cannbotir-verify-target-buffer-sync` 的误报；kernel trace 期对该形态直接
发射直写形态（数值语义相同）。

## 4. 用法

`MoonepBuffer` 路径：

```python
plan = buf.plan(tpe, topk)                 # tpe [E]、topk [S,K]，int32
dst_n, cu_seqlens, etc, src_n, vtopk = plan
```

裸 API 路径：

```python
from moonep_context import create_moonep_context
from moonep_plan import moonep_plan

ctx = create_moonep_context(group=dist.group.WORLD)
topk = topk_idx.to(torch.int32).npu()                            # [S, K]
tpe = torch.bincount(topk.cpu().flatten(), minlength=E).to(torch.int32).npu()
dst_n, cu_seqlens, copy_list, src_n, vtopk = moonep_plan(ctx, tpe, topk)
```

裸 API 的 `create_moonep_context` 缺省只分配 plan 交换区下界；只调 plan 时
足够，后续要跑 `prefetch_weight` / `reduce_grad` 时须按 `_workspace_need()`
显式传入更大的 workspace（见 [MoonEPBuffer.md](MoonEPBuffer.md) 与
[prefetch_weight.md](prefetch_weight.md)）。
