# make_layout

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

根据 `shape` 指定的形状和 `stride` 指定的跨度创建只读的 Layout 对象。省略 `stride` 时，按行主序推导连续跨度。

## 函数原型

```python
make_layout(shape: Shape, *, stride: Stride | None=None) -> Layout
```

## 参数说明

| 参数名 | <span style="white-space: nowrap">输入/输出</span> | 说明 |
| --- | --- | --- |
| `shape` | 输入 | <p>必填。布局的逻辑形状，可以是以下类型：</p> <ul> <li><code>int</code>：一维静态非负整数长度。</li> <li><code>Kernel 运行时整数标量</code>：运行时维度长度，须为非负整数。</li> <li><code>tuple 或 list</code>：指定各维长度，可嵌套；静态嵌套形状须显式指定 stride。</li> </ul>  |
| `stride` | 输入 | <p>各维坐标增加 1 时的线性元素偏移增量，单位为元素，可以是以下类型：</p> <ul> <li><code>None</code>：默认值；按行主序推导连续跨度，末维跨度为 1；支持平坦静态形状和含运行时长度的形状。静态嵌套形状须显式指定 stride。</li> <li><code>int</code>：一维静态非负整数跨度。</li> <li><code>Kernel 运行时整数标量</code>：运行时跨度，须为非负整数。</li> <li><code>tuple 或 list</code>：逐维指定跨度，可嵌套，结构须与 shape 一致。</li> </ul>  |

## 返回值说明

返回记录形状和跨度的 [Layout](overview.md) 对象，可通过 `shape`、`stride` 属性读取。

## 约束说明

无

## 调用示例

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def example_kernel(y: cbd.Tensor):
    layout = cbd.make_layout((2, 3))
    other = cbd.make_layout((2, 3), stride=(1, 2))
    y[0] = layout.shape[0]
    y[1] = layout.shape[1]
    y[2] = layout.stride[0]
    y[3] = layout.stride[1]
    y[4] = other.stride[0]
    y[5] = other.stride[1]


@cbd.host
def run(y):
    example_kernel[1](y)


y = torch.empty(6, dtype=torch.int64, device="npu")
run(y)
torch.npu.synchronize()
print("shape:", y[:2].cpu().tolist())
print("默认 stride:", y[2:4].cpu().tolist())
print("显式 stride:", y[4:].cpu().tolist())
```

输入：

```text
shape=(2, 3)
stride=None、(1, 2)
```

输出：

```text
shape: [2, 3]
默认 stride: [3, 1]
显式 stride: [1, 2]
```
