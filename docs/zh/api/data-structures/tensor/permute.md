# Tensor.permute

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

按照参数`dims`指定的顺序重新排列Tensor的逻辑轴，返回新的Tensor视图。本操作不搬运或重排实际数据。

## 函数原型

```python
permute(self, *dims) -> Tensor
```

## 参数说明

| 参数名 | <span style="white-space: nowrap">输入/输出</span> | 说明 |
| --- | --- | --- |
| `dims` | 输入 | <p>必填。输出各轴对应的输入轴顺序，可以是以下类型：</p> <ul> <li><code>多个 int 位置参数</code>：逐个指定输入轴编号。</li> <li><code>tuple[int, ...]</code>：用元组指定完整轴排列。</li> <li><code>list[int]</code>：与 tuple 的排列规则相同。</li> </ul> <p>每个输入轴必须且只能出现一次；负数轴编号从末轴向前计数，<code>-1</code> 表示最后一轴。</p>   |

## 返回值说明

返回按 `dims` 排列逻辑轴的新 Tensor，共享原存储，`dtype`、`rank` 和 `memloc` 不变。

例如，普通ND Tensor的形状为`(2, 3)`、跨度为`(3, 1)`时，`permute(1, 0)`返回形状为`(3, 2)`、跨度为`(1, 3)`的视图。交换二维NZ、ZN Tensor的两个轴时，`data_format`在`"nz"`和`"zn"`之间切换。对于存储在L1中的Tensor，返回视图保留原`physical_layout`。

## 约束说明

- 对已不支持读取`stride`和`data_format`的变换视图调用本接口后，仍不支持读取这两个属性；可以通过`physical_layout`查看原数据的存储布局。相关示例见[stride](./properties/stride.md#形状变换后的跨度)。
- 逻辑形状不能包含嵌套 tuple。
- 支持 GM 和片上存储中的 Tensor。

## 调用示例

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def example_kernel(y: cbd.Tensor):
    t = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((4, 8)))
    u = t.permute(1, 0)
    y[0] = u.shape[0]
    y[1] = u.shape[1]


@cbd.host
def run(y):
    example_kernel[1](y)


y = torch.empty(2, dtype=torch.int64, device="npu")
run(y)
torch.npu.synchronize()
print("shape:", y.cpu().tolist())
```

输入：

```text
原 shape=(4, 8), dtype=float32
t.permute(1, 0)
```

输出：

```text
shape: [8, 4]
```
