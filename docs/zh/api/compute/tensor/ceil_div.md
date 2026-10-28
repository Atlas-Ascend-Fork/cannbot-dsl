# ceil_div

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将输入形状转换为以切块为单位的形状，各维度的值表示该维需要的块数，不足一个完整块的部分按一个块计数。具体操作为，从 `input` 的最后一个叶维开始，与 `tiler` 中对应的块长逐项相除并向上取整，得到以切块为单位的形状，没有参与切块的前部维度保持不变，结果保留 `input` 的嵌套结构。

## 函数原型

```python
ceil_div(input: Shape, tiler: Tiler) -> Shape
```

## 参数说明

| 参数名 | <span style="white-space: nowrap">输入/输出</span> | 说明 |
| --- | --- | --- |
| `input` | 输入 | <p>必填。输入形状，描述各维度的长度。可以是以下类型：</p> <ul><li><code>int</code>：一维静态长度。</li><li><code>Kernel 运行时整数值</code>：一维运行时长度。</li><li><code>tuple 或 list</code>：组合多个维度，可嵌套；与 <code>tiler</code> 从右侧按最内层维度配对，未配对的前部维度保持不变。</li></ul> |
| `tiler` | 输入 | <p>必填。切块大小，与 <code>input</code> 从最后一个叶维开始逐项对应。可以是以下类型：</p> <ul><li><code>int</code>：一维静态正整数块长。</li><li><code>Kernel 运行时整数值</code>：一维运行时正整数块长。</li><li><code>tuple 或 list</code>：组合各最内层维度的正整数块长。</li></ul><p>最内层维度的数量不得超过 <code>input</code>；每对维度按 <code>(a + b - 1) // b</code> 计算，结果保留 <code>input</code> 的嵌套结构。</p> |

## 返回值说明

返回保持 input 嵌套结构的 Shape；全静态输入在追踪期折叠为 Python 值，动态输入
在 Kernel 执行时计算。

## 约束说明

无。

## 调用示例

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def example_kernel(y: cbd.Tensor):
    counts = cbd.ceil_div((10, 6, 7), (3, 4))
    y[0] = counts[0]
    y[1] = counts[1]
    y[2] = counts[2]


@cbd.host
def run(y):
    example_kernel[1](y)


y = torch.empty(3, dtype=torch.int64, device="npu")
run(y)
torch.npu.synchronize()
print(y.cpu().tolist())
```

输入：

```text
input=(10, 6, 7), tiler=(3, 4)
```

输出：

```text
[10, 2, 2]
```
