# MoonEP 算子

`moonep_plan` 用于专家并行场景的均衡规划，`prefetch_weight` 用于跨卡权重搬运，
将 home rank 的专家权重传输到消费 rank 的对称预取槽。配套模块提供通信上下文、
显存视图和显存统计。设计参考 [MoonEP](https://github.com/MoonshotAI/MoonEP)。

## 规划（moonep_plan）

`moonep_plan(context, tokens_per_expert, topk_experts_sk)` 由本 rank 的路由计数
与 top-k 索引，产出规划五件套（纯张量，跨调用、跨前后向复用）：跨卡交换计数
后在 demand 矩阵上做 cap 均衡贪心（一卡一源），生成副本清单与虚拟专家投影，
供 `prefetch_weight` 与 ep dispatch/combine 复用链消费。

```python
from moonep_context import create_moonep_context
from moonep_plan import moonep_plan

context = create_moonep_context()
dst_n, cu_seqlens, experts_to_copy, src_n, vtopk = moonep_plan(
    context, tokens_per_expert, topk_experts_sk,
)
```

| 参数 / 返回 | shape | dtype | 说明 |
| :--- | :--- | :--- | :--- |
| `tokens_per_expert` | `[E]` | `torch.int32` | 本 rank 各专家 token 计数 |
| `topk_experts_sk` | `[S, K]` | `torch.int32` | top-k 专家索引 |
| `cu_seqlens` | `[2*epn]` | `torch.int32` | 分组 GEMM 分段偏移（本地专家 + 预取槽，均按专家 id 升序） |
| `experts_to_copy` | `[R, epn]` | `torch.int32` | 各 rank 需预取的远程专家清单，空槽 -1；每行至多一个远程 home rank |
| `vtopk` | `[N]` | `torch.int32` | 虚拟专家 id（扩展空间 `E' = 2E`），供 ep 复用链路由改写 |
| `dst_n` / `src_n` | `[N]` | `torch.int32` | 零占位（vtopk-only 快路径不计算） |

形状约束：`E <= 1024`、`R <= 64`、`E % R == 0`；`S*K` 任意规模原生整除
（分块粒度降档到 N 的因子，无尾补机制）。

## 权重预取（prefetch_weight）

`prefetch_weight(context, experts_to_copy, real_weights, local_weights, epn)`
按调用者提供的专家清单，把远程 home rank 的专家权重写入本地预取槽。
`real_weights` 和 `local_weights` 均为三个 BF16 权重张量组成的列表或元组；
每组真实权重形状为 `[epn, W]`，本地视图形状为 `[2*epn, W]`。
`experts_to_copy` 为设备端 int32 张量 `[rank_size, epn]`，空槽填 `-1`；
每个目标 rank 的非空槽至多来自同一个 home rank。

通信上下文使用独立 HCCL 通信器，不能与其他占用同一对称堆的算子共用。
规划算子调用后自动设置 `context.last_plan_shape`；独立使用预取时调用者
须自行设置该字段。

```python
from moonep_context import create_moonep_context
from prefetch_weight import create_weight_view, prefetch_weight

context = create_moonep_context()
context.last_plan_shape = (num_tokens, top_k)
views = [
    create_weight_view(context, epn, w_elems, group=g, groups=3)
    for g in range(3)
]
prefetch_weight(context, experts_to_copy, real_weights, views, epn)
```

在仓库根目录设置 `PYTHONPATH="$PWD/samples/mc2/moonep:$PYTHONPATH"`。
接口支持多槽视图。`async_finish=True` 返回完成事件，调用方通过事件等待汇合。

## 验证

需要匹配的 CANN、PyTorch/torch_npu、HCCL 和至少两张支持设备。

规划算子验收（前向 E2E 对 golden，另含布局公式与启动块数约束自检）：

```bash
python -m torch.distributed.run --nproc_per_node=8 -m pytest \
    test/mc2/moonep/test_moonep_plan.py -q
```

预取算子独立验收检查数据正确性及 Buffer 的 `(mem_loc, dtype, layout)` 调用：

```bash
python -m torch.distributed.run --nproc_per_node=4 -m pytest \
    test/mc2/moonep/test_prefetch_weight.py -q
```

设备性能基准见
[bench_prefetch_weight.py](../../../test/mc2/moonep/bench_prefetch_weight.py)，
可配置输入形状、预热和采样次数。
