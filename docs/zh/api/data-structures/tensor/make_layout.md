# make_layout

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

根据`shape`指定的形状和`stride`指定的跨度创建Layout对象。省略`stride`时，按行主序推导连续跨度，例如形状`(2, 3)`对应跨度`(3, 1)`。

## 函数原型

```python
make_layout(shape: Shape, *, stride: Stride | None = None) -> Layout
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `shape` | 输入 | 必填，各维的长度。一个整数表示一维长度，如`8`；元组或列表列出各维长度，如`(2, 3)`或`[2, 3]`。长度可以直接写成整数，也可以使用从输入Tensor读取的整数，各维长度须非负。 |
| `stride` | 输入 | 各维坐标增加1时的元素偏移量，单位为元素。默认`None`，根据`shape`按行连续排列。显式指定时，使用与`shape`结构一致的非负整数、元组或列表；也可以使用从输入Tensor读取的整数作为跨度。 |

### 固定长度和跨度

以下代码中的`cbd`为通过`import cannbotdsl as cbd`导入的模块别名。

| 写法 | 形状 | 跨度 | 含义 |
| --- | --- | --- | --- |
| `cbd.make_layout(8)` | `8` | `1` | 一维，包含8个连续元素。 |
| `cbd.make_layout((2, 3))` | `(2, 3)` | `(3, 1)` | 2行、3列，各行连续存放。 |
| `cbd.make_layout([2, 3])` | `(2, 3)` | `(3, 1)` | 与使用元组指定形状相同。 |
| `cbd.make_layout((2, 3), stride=(8, 1))` | `(2, 3)` | `(8, 1)` | 相邻两行的起点相隔8个元素，同一行的相邻元素相隔1个元素。 |
| `cbd.make_layout(((2, 3), 4), stride=((12, 4), 1))` | `((2, 3), 4)` | `((12, 4), 1)` | 形状包含嵌套结构，跨度按相同结构逐项指定。 |

### 使用输入数据给出的长度和跨度

例如，`sizes`是一维整数Tensor，依次存放行数、列数和行跨度。在Kernel中读取前两项后，可以将其作为维度长度：

```python
n = sizes[0]
m = sizes[1]
layout = cbd.make_layout((n, m))
```

当输入的前两项为`[4, 8]`时，布局形状为`(4, 8)`；改为`[6, 8]`时，形状为`(6, 8)`。布局尺寸由执行时读取的输入数据决定。

跨度也可以由输入给出。例如，`pitch = sizes[2]`后使用`stride=(pitch, 1)`；若`pitch`为16，相邻两行的起点就相隔16个元素。

## 返回值说明

返回记录形状和跨度的Layout对象，可通过`layout.shape`和`layout.stride`读取。相关概念见[shape / rank](./properties/shape-and-rank.md)和[stride](./properties/stride.md)。

## 约束说明

- 各维长度均为固定整数的嵌套形状不能省略`stride`，例如`((2, 3), 4)`须显式给出`((12, 4), 1)`等匹配结构的跨度。
- `stride`须使用关键字传入，例如`make_layout((2, 3), stride=(8, 1))`。

## 调用示例

以下示例从输入Tensor读取行数、列数和行跨度，分别创建自动推导跨度及显式指定跨度的布局，再读取其属性并校验。

```python
import torch
import torch_npu  # noqa: F401
import cannbotdsl as cbd


@cbd.kernel
def example_kernel(sizes: cbd.Tensor, out: cbd.Tensor):
    n = sizes[0]
    m = sizes[1]
    pitch = sizes[2]
    automatic = cbd.make_layout((n, m))
    explicit = cbd.make_layout((n, m), stride=(pitch, 1))
    out[0] = automatic.shape[0]
    out[1] = automatic.shape[1]
    out[2] = automatic.stride[0]
    out[3] = automatic.stride[1]
    out[4] = explicit.stride[0]
    out[5] = explicit.stride[1]


@cbd.host
def run(sizes, out):
    example_kernel[1](sizes, out)


def main():
    for values in ((2, 3, 8), (4, 5, 16)):
        sizes = torch.tensor(values, dtype=torch.int64, device="npu")
        out = torch.empty(6, dtype=torch.int64, device="npu")
        run(sizes, out)
        result = out.cpu().tolist()
        n, m, pitch = values
        assert result == [n, m, m, 1, pitch, 1]
        print("输入:", list(values))
        print("shape:", result[:2])
        print("默认 stride:", result[2:4])
        print("显式 stride:", result[4:])


if __name__ == "__main__":
    main()
```

预期结果：

```text
输入: [2, 3, 8]
shape: [2, 3]
默认 stride: [3, 1]
显式 stride: [8, 1]
输入: [4, 5, 16]
shape: [4, 5]
默认 stride: [5, 1]
显式 stride: [16, 1]
```
