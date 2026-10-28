# Dim
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

`Dim` 用于在 [TensorSpec](./tensor_spec.md) 和 [TensorListSpec](./tensor_list_spec.md) 中表示运行时才能确定的长度，并设置该长度的取值范围和整除要求。支持通过 `+`、`-`、`*` 和 `//` 表达维度之间的关系，例如用 `M * 2` 表示某一维的长度是 `M` 的两倍。

## 函数原型

```python
Dim(
    name: str,
    min: int = 1,
    max: int | None = None,
    multiple_of: int = 1,
) -> Dim
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `name` | 输入 | 维度名称，为非空字符串，例如 `"M"` 或 `"length"`。 |
| `min` | 输入 | 长度下限，包含该值。取值范围为 `[0, 2**63 - 1]`，默认值为 `1`。 |
| `max` | 输入 | 长度上限，包含该值。取值范围为 `[min, 2**63 - 1]`；默认值为 `None`，表示不设置上限。 |
| `multiple_of` | 输入 | 长度必须是该值的整数倍，并满足上下限要求。取值范围为 `[1, 2**63 - 1]`，默认值为 `1`。例如设为 `8`，表示长度必须是 `8` 的整数倍。 |

## 返回值说明

返回一个表示动态维度的 `Dim` 对象。

## 约束说明

- 多个普通Tensor参数使用同名 `Dim` 时，须设置相同的上下限和整除要求，运行时对应长度也须相同。需要表示不同长度时，请使用不同名称。

## 调用示例

以下示例用Dim声明动态长度，并构造第二维为第一维两倍的Tensor规格。完整的编译和运行流程见 [TensorSpec](./tensor_spec.md)。

```python
import cannbotdsl as cbd

length = cbd.Dim("length", min=8, max=32, multiple_of=8)
spec = cbd.TensorSpec((length, length * 2), cbd.dtypes.float32)

print(spec.shape)
```

预期结果：

```text
(Dim('length', min=8, max=32, multiple_of=8), (2 * Dim('length', min=8, max=32, multiple_of=8)))
```
