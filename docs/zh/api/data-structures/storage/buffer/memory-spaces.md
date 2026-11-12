# 片上存储空间

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

通过存储空间的`view`接口获取完整的字节视图，用于手动选择缓冲区所在的字节区间。选取区域并指定元素类型和形状后，使用[make_buffer](./make_buffer.md)绑定为可参与搬运、计算的Tensor；多槽缓冲区使用[make_ring_buffer](../ring-buffer/make_ring_buffer.md)。

需要由框架安排存储地址时，使用[Buffer](./Buffer.md)分配片上Tensor。各存储位置的含义见[memloc](../../tensor/properties/memloc.md)。

## 函数原型

```python
UB.view(*shape: int, dtype: DType | None = None) -> Tensor
L1.view(*shape: int, dtype: DType | None = None) -> Tensor
L0A.view(*shape: int, dtype: DType | None = None) -> Tensor
L0B.view(*shape: int, dtype: DType | None = None) -> Tensor
L0C.view(*shape: int, dtype: DType | None = None) -> Tensor
BIAS.view(*shape: int, dtype: DType | None = None) -> Tensor
SSBUF.view(*shape: int, dtype: DType | None = None) -> Tensor
FBUF.view(*shape: int, dtype: DType | None = None) -> Tensor
FBUF_ACT.view(*shape: int, dtype: DType | None = None) -> Tensor
```

上述入口均由`cannbotdsl`提供。通过`import cannbotdsl as cbd`导入后，可按下表调用：

| 空间 | 完整容量（字节） | 调用示例 |
| --- | --- | --- |
| UB | 262144 | `cbd.UB.view(262144)` |
| L1 | 524288 | `cbd.L1.view(524288)` |
| L0A | 65536 | `cbd.L0A.view(65536)` |
| L0B | 65536 | `cbd.L0B.view(65536)` |
| L0C | 262144 | `cbd.L0C.view(262144)` |
| BIAS | 4096 | `cbd.BIAS.view(4096)` |
| SSBUF | 3072 | `cbd.SSBUF.view(3072)` |
| FBUF | 4096 | `cbd.FBUF.view(4096)` |
| [FBUF_ACT](../fbuf-act-view.md) | 2048 | `cbd.FBUF_ACT.view(2048)` |

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `shape` | 输入 | 完整空间视图的形状。传入一个或多个编译期已知的正整数，或一个正整数元组、列表。元素数乘积必须等于上表中对应空间的完整字节容量。例如，UB可使用`view(262144)`或`view(512, 512)`。选取局部区域时，在返回视图上进行字节切片。 |
| `dtype` | 输入 | 默认`None`，返回`uint8`视图；显式指定时也只能为`dtypes.uint8`。需要其他元素类型时，先选取字节区间，再使用[Tensor.view](../../tensor/view.md)指定类型和形状。 |

## 返回值说明

返回覆盖完整存储空间的`uint8` Tensor，形状为`shape`，`memloc`为对应的存储位置。同一Kernel中重复获取同一空间的视图时，共享底层存储。接口不初始化存储内容。

## 约束说明

- 在Kernel函数中使用。
- 同一Kernel的同一存储空间不能混用手动空间视图与`Buffer`自动分配；不同空间可以分别选择方式。
- 传给`make_buffer`的切片区域必须非空，字节偏移须满足`0 ≤ start < stop ≤ 完整容量`。具体搬运的地址对齐要求见[mem_copy](../../../data-movement/mem-copy.md)。
- SSBUF视图覆盖完整的3072字节空间；非MIX模式下，各核的访问仍受各自1024字节范围限制。
- SSBUF、FBUF和FBUF_ACT不支持绑定为RingBuffer。

## 调用示例

以下示例从UB中选择128字节的区域，创建形状为`(4, 8)`的`float32`缓冲区，完成输入到输出的数据搬运。

```python
import torch
import torch_npu  # noqa: F401
import cannbotdsl as cbd


@cbd.kernel
def copy_kernel(x: cbd.Tensor, out: cbd.Tensor):
    space = cbd.UB.view(262144)
    region = space[128:256,]
    typed = region.view(dtype=cbd.dtypes.float32).view(4, 8)
    temp = cbd.make_buffer(typed)
    cbd.mem_copy(temp, x)
    cbd.mem_copy(out, temp)


@cbd.host
def run(x, out):
    copy_kernel[1](x, out)


def main():
    x = torch.arange(32, dtype=torch.float32, device="npu").reshape(4, 8)
    out = torch.empty_like(x)
    run(x, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), x.cpu())
    print("数据搬运结果一致")


if __name__ == "__main__":
    main()
```

预期结果：

```text
数据搬运结果一致
```
