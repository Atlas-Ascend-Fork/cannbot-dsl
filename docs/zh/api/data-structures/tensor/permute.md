# Tensor.permute

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

按照参数 `dims` 指定的顺序重新排列当前 Tensor 的逻辑轴，返回按该顺序排列形状和跨度的新 Tensor 视图。本操作不立即搬运或重排实际数据。

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

普通 ND 布局的逻辑形状、跨度和物理布局均按 `dims` 排列。交换二维 NZ/ZN Tensor 的两个轴时，格式在 NZ 和 ZN 之间切换。对于存储在 L1 中的 Tensor，返回视图保留原物理布局。原 Tensor 为组合视图时，返回结果仍为组合视图，其 `physical_layout` 保持不变，由编译器根据 `dims` 调整坐标映射。

## 约束说明

- 返回组合视图时，其 `stride`、`physical_format` 和 `data_format` 不支持读取，读取会报错；`physical_layout`、`physical_shape` 和 `physical_stride` 仍可读取。

- 逻辑形状不能包含嵌套 tuple。
- 支持交换二维 NZ/ZN Tensor 的两个轴，布局格式在 NZ 与 ZN 之间切换，底层数据不重排。
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
