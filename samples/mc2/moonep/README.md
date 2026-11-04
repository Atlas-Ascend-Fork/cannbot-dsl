# MoonEP 权重预取

`prefetch_weight` 用于专家并行场景中的跨卡权重搬运，将 home rank 的专家权重
传输到消费 rank 的对称预取槽。配套模块提供通信上下文、显存视图和显存统计。
设计参考 [MoonEP](https://github.com/MoonshotAI/MoonEP)。

## 接口与调用约定

`prefetch_weight(context, experts_to_copy, real_weights, local_weights, epn)`
按调用者提供的专家清单，把远程 home rank 的专家权重写入本地预取槽。
`real_weights` 和 `local_weights` 均为三个 BF16 权重张量组成的列表或元组；
每组真实权重形状为 `[epn, W]`，本地视图形状为 `[2*epn, W]`。
`experts_to_copy` 为设备端 int32 张量 `[rank_size, epn]`，空槽填 `-1`；
每个目标 rank 的非空槽至多来自同一个 home rank。

通信上下文使用独立 HCCL 通信器，不能与其他占用同一对称堆的算子共用。
创建权重视图前，调用者须设置 `context.last_plan_shape = (num_tokens, top_k)`；
该字段及专家清单由调用者准备。

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
独立验收检查数据正确性及 Buffer 的 `(mem_loc, dtype, layout)` 调用：

```bash
python -m torch.distributed.run --nproc_per_node=4 -m pytest \
    test/mc2/moonep/test_prefetch_weight.py -q
```

设备性能基准见
[bench_prefetch_weight.py](../../../test/mc2/moonep/bench_prefetch_weight.py)，
可配置输入形状、预热和采样次数。
