# Tensor.pad_to

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将 Tensor 的逻辑形状扩展到目标形状，新增位置使用 `value` 指定的值，原范围内的元素保持不变。可以增大一个或多个现有轴的长度，新增位置均位于对应轴的末尾；不能新增轴，也不能缩小原有轴。

例如，原形状为 `(2, 3)` 时：

- 目标形状为 `(2, 5)`：每行末尾增加两列。
- 目标形状为 `(4, 3)`：在末尾增加两行。
- 目标形状为 `(4, 5)`：同时增加行和列。

原来的左上角 `2×3` 区域保持不变，超出原范围的位置均使用 `value` 指定的值。

## 函数原型

```python
pad_to(self, shape, *, value) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| `shape` | 输入 | 目标逻辑形状。必填，各维长度可以是 Python 整数或 Kernel 运行时整数值。 |
| `value` | 输入 | 新增位置的填充值。必填，为 Python 整数、浮点数或 Kernel 运行时标量。 |

## 返回值说明

返回目标逻辑形状的 Tensor 视图，`dtype` 与原 Tensor 相同。该视图包含填充值的访问规则，不能将新增位置理解为原存储中已存在的元素。

## 约束说明

- `shape` 的维数须等于 `self.rank`，各维长度不得小于原维度长度。
- 各维长度须位于 `0` 至 `2^63 - 1` 范围内；运行时长度也须满足上述要求。

## 调用示例

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def example_kernel(x: cbd.Tensor, y: cbd.Tensor):
    result = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float16, cbd.make_layout((2, 16)))
    cbd.mem_copy(result, x.pad_to((2, 16), value=-7.0))
    cbd.mem_copy(y, result.flatten())


@cbd.host
def run(x, y):
    example_kernel[1](x, y)


x = torch.arange(20, dtype=torch.float16, device="npu").reshape(2, 10)
y = torch.empty(32, dtype=torch.float16, device="npu")
run(x, y)
torch.npu.synchronize()
print(y.cpu().reshape(2, 16).tolist())
```

输入：

```text
x=arange(20).reshape(2, 10), dtype=float16
目标 shape=(2, 16), value=-7.0
```

输出：

```text
[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, -7.0, -7.0, -7.0, -7.0, -7.0, -7.0], [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, -7.0, -7.0, -7.0, -7.0, -7.0, -7.0]]
```
