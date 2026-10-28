# Tensor.broadcast_to

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

根据 `shape` 指定的目标形状广播当前 Tensor，返回新的 Tensor 视图。支持以下两种扩展方式，可同时使用：

- 扩展原有轴：将原形状中长度为 `1` 的任意轴扩展到目标长度，不限于最外层轴。
- 新增轴：在原形状的最前面增加一个或多个轴，各新增轴的长度由目标形状指定。

原有轴与目标形状从最后一轴开始对应；原有轴的长度不为 `1` 时，对应的目标长度必须保持不变。沿扩展轴或新增轴访问不同位置时，访问的是原 Tensor 中的同一份数据。

## 函数原型

```python
broadcast_to(self, shape) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| `shape` | 输入 | 目标形状。必填，为静态正整数序列，每项不超过 `2^63-1`。 |

## 返回值说明

返回目标形状的 Tensor，保留原元素类型。普通 ND 的新增轴和扩展轴跨度为 0；原 Tensor 为 NZ / ZN 布局或组合视图时，返回组合视图，保留原物理布局，由编译器根据广播规则维护坐标映射，访问原数据。

## 约束说明

- 原 Tensor 为 NZ / ZN 布局或组合视图时，返回组合视图；普通 ND 布局的广播不因此变为组合视图。
- 返回组合视图时，其 `stride`、`physical_format` 和 `data_format` 不支持读取，读取会报错；`physical_layout`、`physical_shape` 和 `physical_stride` 仍可读取。

- 目标维数不得小于 `self.rank`。

## 调用示例

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def example_kernel(y: cbd.Tensor):
    t = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((1, 8)))
    u = t.broadcast_to((4, 8))
    y[0] = u.shape[0]
    y[1] = u.shape[1]
    y[2] = u.stride[0]
    y[3] = u.stride[1]


@cbd.host
def run(y):
    example_kernel[1](y)


y = torch.empty(4, dtype=torch.int64, device="npu")
run(y)
torch.npu.synchronize()
print("shape:", y[:2].cpu().tolist())
print("stride:", y[2:].cpu().tolist())
```

输入：

```text
原 shape=(1, 8), 目标 shape=(4, 8)
```

输出：

```text
shape: [4, 8]
stride: [0, 1]
```
