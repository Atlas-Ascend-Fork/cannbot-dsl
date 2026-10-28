# tile_slice
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

在按块搬运或计算时，按tile坐标取得输入Tensor的子块视图。返回视图与输入共享存储，不复制数据。

## 函数原型

```python
tile_slice(input: Tensor, tiler: Tiler, coord: Coord) -> Tensor
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `input` | 输入 | 输入[Tensor](../../data-structures/tensor/overview.md)。<br>动态tile仅支持Identity/ND布局的GM Tensor。 |
| `tiler` | 输入 | tile大小，单位为元素个数，支持整数、整数序列或[Tiler](../../data-structures/tiler/overview.md)。<br>- 静态大小：使用正整数或正整数序列指定。<br>- 动态大小：使用[make_tiler](../../data-structures/tiler/make_tiler.md)创建的`Tiler`指定；通过`alignment`参数声明对齐要求。 |
| `coord` | 输入 | 非负tile序号，支持整数、整数序列或[Coord](../../data-structures/layout/coord.md)。例如，`tiler=(64, 128)`、`coord=(2, 1)`对应起点`(128, 128)`。<br>尾块不足`tiler`指定的大小时，按输入剩余大小返回；某一维度的起点达到或超出输入范围时，该维度大小为0。 |


## 返回值说明

返回与输入共享存储的`Tensor`视图。`tiler`为K维时，返回视图对应输入Shape的最后K个逻辑维度，并保留这些维度的步长（stride）。

## 约束说明

- `tiler`和`coord`展开后的维度数必须相同；tile维度数不能超过输入维度数。
- 即使返回视图的某个维度大小为0，后续访问时计算的地址也不能超出目标指针可表示的范围。
- 视图与原Tensor共享数据。

## 调用示例

以下示例取得输入Tensor的第二个tile，并在Host侧校验输出为原输入的后16个元素。

```python
import cannbotdsl as cb
import torch
import torch_npu

@cb.kernel
def copy_second_tile_kernel(src: cb.Tensor, dst: cb.Tensor):
    # 按tile序号取得第二个子块，共享输入存储。
    src_tile = cb.tile_slice(src, (16,), (1,))
    tmp = cb.RingBuffer(
        cb.MemLoc.UB,
        dtype=src.dtype,
        layout=cb.make_layout((16,)),
        depth=1,
    )

    slot = tmp.next_write()
    cb.mem_copy(slot, src_tile)
    slot = tmp.next_read()
    cb.mem_copy(dst, slot)

@cb.host
def copy_second_tile(src: cb.Tensor, dst: cb.Tensor):
    copy_second_tile_kernel[1](src, dst)

def main():
    src = torch.arange(32, dtype=torch.float32, device="npu:0")
    dst = torch.empty(16, dtype=torch.float32, device="npu:0")

    copy_second_tile(src, dst)
    torch.npu.synchronize()

    actual = dst.cpu()
    expected = torch.arange(16, 32, dtype=torch.float32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    print("output:", actual.tolist())

if __name__ == "__main__":
    main()
```

预期结果：

```text
output: [16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0]
```
