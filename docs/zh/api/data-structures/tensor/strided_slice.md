# Tensor.strided_slice

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

按各轴起点、选取数量和步长派生子视图。第 i 轴选择 `starts[i] + j * steps[i]` 对应的位置，其中 `0 ≤ j < sizes[i]`。

## 函数原型

```python
strided_slice(self, starts, sizes, steps=None) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| `starts` | 输入 | 各轴起点。必填，序列长度为 `self.rank`，各项为非负整数或运行时整数。 |
| `sizes` | 输入 | 各轴选取数量。必填，序列长度为 `self.rank`，各项为正整数或运行时整数。 |
| `steps` | 输入 | 各轴选取步长。默认 `None`，全部为 1；也可传入长度为 `self.rank` 的正整数或运行时整数序列。 |

## 返回值说明

返回形状为 `sizes` 的 Tensor，保留元素类型和存储层级。普通 ND 逻辑跨度为源跨度乘对应步长；原 Tensor 为 NZ / ZN 布局或组合视图时，返回组合视图，保留原物理布局，由编译器根据切片起点和步长维护坐标映射，访问原数据。

## 约束说明

- 原 Tensor 为 NZ / ZN 布局或组合视图时，返回组合视图；普通 ND 布局的步长切片不因此变为组合视图。
- 返回组合视图时，其 `stride`、`physical_format` 和 `data_format` 不支持读取，读取会报错；`physical_layout`、`physical_shape` 和 `physical_stride` 仍可读取。

- 各轴须满足 `starts[i] + (sizes[i] - 1) * steps[i] < self.shape[i]`；运行时参数由调用者保证有效。

## 调用示例

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def example_kernel(y: cbd.Tensor):
    t = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((4, 8)))
    u = t.strided_slice((0, 1), (2, 3), (2, 2))
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
t.strided_slice((0, 1), (2, 3), (2, 2))
```

输出：

```text
shape: [2, 3]
```
