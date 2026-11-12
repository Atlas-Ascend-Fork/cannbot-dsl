# make_tiler

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将参数 `tiler` 指定的可嵌套的切块大小，与 `alignment` 指定的可选对齐条件组合为可复用的切块对象。`alignment` 的对齐要求不作用于尾块。

## 函数原型

```python
make_tiler(tiler: Tiler, *, alignment=None)
```

## 参数说明

| 参数名 | <span style="white-space: nowrap">输入/输出</span> | 说明 |
| --- | --- | --- |
| <span id="tiler-参数可选值"></span>`tiler` | 输入 | <p>必填。可嵌套的切块大小，可以是以下类型：</p> <ul> <li><code>int</code>：一维静态正整数块长。</li> <li><code>核函数中计算得到的整数值</code>：例如输入Tensor的某一维长度；作为块长时须为64位正整数。</li> <li><code>tuple 或 list</code>：组合各维块长，可嵌套；保留各维的嵌套分组，维度顺序见下例。</li>  </ul>   |
| <span id="alignment-参数可选值"></span>`alignment` | 输入 | <p>指定块长的对齐要求，可以是以下类型：</p> <ul> <li><code>None</code>：默认值，不额外声明对齐条件。</li> <li><code>int</code>：编译期正整数，对所有最内层维度采用同一对齐要求。</li> <li><code>tuple[int, ...]</code>：各项为编译期正整数；单元素时对所有最内层维度采用同一要求，多元素时须按从左到右的顺序，与各最内层维度一一对应。</li> <li><code>list[int]</code>：与tuple的逐维对应规则相同。</li> </ul>  |

例如，`tiler=((16, 32), 8)`包含三个最内层维度，按顺序为16、32、8；`alignment=(8, 16, 4)`分别指定这三个维度的对齐要求。

## 返回值说明

返回记录切块大小和对齐条件的切块对象，可传入 `tile_slice()`。

## 约束说明

- 各维块长须为正整数，`alignment`各项须为编译时确定的正整数。
- 每个块长须能被对应对齐值整除。例如，`alignment=8`时，块长可为16或24，不能为18；块长由Tensor数据等运行时值决定时，接口不自动检查整除关系，调用方须保证取值满足要求。

## 调用示例

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def example_kernel(x: cbd.Tensor, y: cbd.Tensor):
    tiler = cbd.make_tiler((8, 8), alignment=8)
    tile = cbd.tile_slice(x, tiler, (1, 1))
    y[0] = tile.shape[0]
    y[1] = tile.shape[1]
    y[2] = tile[0, 0]
    y[3] = tile[1, 3]


@cbd.host
def run(x, y):
    example_kernel[1](x, y)


x = torch.arange(120, dtype=torch.int64, device="npu").reshape(10, 12)
y = torch.empty(4, dtype=torch.int64, device="npu")
run(x, y)
torch.npu.synchronize()
print("shape:", y[:2].cpu().tolist())
print("两角元素:", y[2:].cpu().tolist())
```

输入：

```text
x=arange(120).reshape(10, 12), tiler=(8, 8), alignment=8, coord=(1, 1)
```

输出：

```text
shape: [2, 4]
两角元素: [104, 119]
```
