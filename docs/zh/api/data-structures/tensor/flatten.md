# Tensor.flatten

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将当前 Tensor 中从 `start_dim` 到 `end_dim` 的逻辑轴合并为一个轴，新轴长度等于这些轴长度的乘积。返回的新 Tensor 视图与当前对象共享存储，不搬运或重排实际数据。

## 函数原型

```python
flatten(self, start_dim=0, end_dim=-1) -> Tensor
```

## 参数说明

| 参数名 | <span style="white-space: nowrap">输入/输出</span> | 说明 |
| --- | --- | --- |
| `start_dim` | 输入 | 合并的起始轴编号，须为编译时确定的整数。默认值为`0`。取值范围为`-self.rank ≤ start_dim < self.rank`；负数从末轴向前计数，`-1`表示最后一轴。 |
| `end_dim` | 输入 | 合并的结束轴编号，包含该轴。须为编译时确定的整数，取值规则与`start_dim`相同。默认值为`-1`。 |

## 返回值说明

返回新 Tensor，逻辑形状将指定连续轴合并，`rank` 随新形状确定；`dtype`、`memloc` 不变，并共享原存储。

起止轴相同时，直接返回原 Tensor。

合并多个轴后，返回视图的`stride`和`data_format`读取规则见[stride](./properties/stride.md#形状变换后的跨度)。

## 约束说明

- Tensor须至少有一个轴，形状须直接列出各轴长度，如`(2, 3)`；不支持`((2, 3), 4)`这样的嵌套形状。
- `end_dim`对应的轴不得位于`start_dim`对应的轴之前。

## 调用示例

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def example_kernel(y: cbd.Tensor):
    t = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((4, 8)))
    u = t.flatten(0, 1)
    y[0] = u.shape[0]


@cbd.host
def run(y):
    example_kernel[1](y)


y = torch.empty(1, dtype=torch.int64, device="npu")
run(y)
torch.npu.synchronize()
print("shape:", y.cpu().tolist())
```

输入：

```text
原 shape=(4, 8), dtype=float32
t.flatten(0, 1)
```

输出：

```text
shape: [32]
```
