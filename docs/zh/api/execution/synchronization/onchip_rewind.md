# onchip_rewind

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将当前 Kernel 中各片上存储层级的地址分配位置重置到起点，使后续创建的 Buffer、RingBuffer 从起始位置按对齐要求重新分配存储，从而复用此前分配过的地址。`reset_sync_id` 控制是否复用已完成同步流程的同步资源，默认为 `True`；设为 `False` 时不启用该复用。

该调用不会清除存储中的数据，也不会销毁已有 Buffer、RingBuffer 或重置 RingBuffer 的读写游标。旧对象及其 Tensor 仍指向原来的存储，后续新分配可能与这些存储部分或全部重叠；新对象写入重叠区域后，旧对象访问该区域时看到的数据也会改变。

地址重新分配不会自动避开仍在使用的存储，该调用也不会等待搬运或计算完成。调用者须保证重叠存储的使用不冲突：新资源使用前，旧资源上的搬运和计算已经完成，旧数据不再需要，且之后不会通过旧对象误读或覆盖新资源的数据。

## 函数原型

```python
onchip_rewind(reset_sync_id: bool=True) -> None
```

## 参数说明

| 参数名 | <span style="white-space: nowrap">输入/输出</span> | 说明 |
| --- | --- | --- |
| `reset_sync_id` | 输入 | <p>是否复用已完成同步流程的同步资源，类型为 Python <code>bool</code>，默认值为 <code>True</code>。可选值：</p><ul><li><code>True</code>：允许复用已完成同步流程的同步资源。</li><li><code>False</code>：不启用上述同步资源复用。</li></ul> |

## 返回值说明

无返回值。

## 约束说明

- 只要当前 Kernel 使用任一[片上存储空间](../../data-structures/storage/on-chip-memory.md)，就不能使用该接口。

## 调用示例

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def example_kernel(y: cbd.Tensor):
    first = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.int64, cbd.make_layout((4,)))
    first[0] = 11
    y[0] = first[0]
    cbd.onchip_rewind(reset_sync_id=False)
    second = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.int64, cbd.make_layout((4,)))
    second[0] = 22
    y[1] = second[0]


@cbd.host
def run(y):
    example_kernel[1](y)


y = torch.empty(2, dtype=torch.int64, device="npu")
run(y)
torch.npu.synchronize()
print(y.cpu().tolist())
```

输入：

```text
两次分配：mem_loc=UB, dtype=int64, shape=(4,)
reset_sync_id=False，先写入并读取 11，再写入并读取 22
```

输出：

```text
[11, 22]
```
