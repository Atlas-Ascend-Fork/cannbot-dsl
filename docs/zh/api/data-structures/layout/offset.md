# Layout.offset

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

计算给定逻辑坐标相对于当前布局绑定起点在内存上的元素偏移。

## 函数原型

```python
offset(self, coordinate) -> int | Int64
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| `coordinate` | 输入 | <p>必填。布局中的逻辑坐标，可以是以下类型：</p><ul><li><code>int</code>：一维静态整数坐标。</li><li><code>Kernel 运行时整数标量</code>：运行时确定的一维坐标。</li><li><code>tuple</code>：按各维顺序指定坐标，各项可包含运行时整数值；嵌套结构须与布局的 <code>shape</code> 一致。</li></ul> |

## 返回值说明

返回给定坐标对应的元素偏移，单位为元素。对于 Tensor 的 `layout`，根据 Tensor 的布局格式和视图变换，计算指定逻辑坐标相对该视图起点的元素偏移。

## 约束说明

- 坐标须位于有效范围，且结构与布局对应。

## 调用示例

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def example_kernel(y: cbd.Tensor):
    layout = cbd.make_layout((4, 8))
    y[0] = layout.offset((1, 2))


@cbd.host
def run(y):
    example_kernel[1](y)


y = torch.empty(1, dtype=torch.int64, device="npu")
run(y)
torch.npu.synchronize()
print("元素偏移:", y.cpu().tolist()[0])
```

输入：

```text
shape=(4, 8), coordinate=(1, 2)
```

输出：

```text
元素偏移: 10
```
