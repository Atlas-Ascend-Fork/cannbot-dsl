# TensorSpec
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

`TensorSpec` 用于描述Tensor参数的形状、数据类型、步长和存储格式，供 `cannbotdsl.compile(...)` 提前编译使用。

## 函数原型

```python
TensorSpec(
    shape: Iterable,
    dtype: DType,
    *,
    stride: Iterable | None = None,
    storage_format: str = "nd",
) -> TensorSpec
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `shape` | 输入 | Tensor各维的长度，支持整数、[Dim](./dim.md) 或维度表达式。静态长度的取值范围为 `[0, 2**63 - 1]`。 |
| `dtype` | 输入 | Tensor的元素类型，支持的取值见 [dtype支持类型](#dtype支持类型)。 |
| `stride` | 输入 | 沿各维移动一个位置时跨过的元素数，项数须与 `shape` 一致。支持整数、`Dim` 或维度表达式；静态值的取值范围为 `[0, 2**63 - 1]`。默认值为 `None`，根据 `shape` 自动推导紧凑行主序步长。 |
| `storage_format` | 输入 | Tensor的存储格式，可选 `"nd"`、`"nz"`，分别表示ND格式和NZ分形格式。默认值为 `"nd"`。 |

### dtype支持类型

| 存储格式 | 支持的数据类型 |
| --- | --- |
| `"nd"` | 浮点类型：`dtypes.float16`、`dtypes.float32`、`dtypes.float64`、`dtypes.bfloat16`、`dtypes.float8_e4m3fn`、`dtypes.float8_e5m2`、`dtypes.float8_e8m0`、`dtypes.hifloat8`、`dtypes.fp4x2_e2m1`<br>有符号整数：`dtypes.int8`、`dtypes.int16`、`dtypes.int32`、`dtypes.int64`<br>无符号整数：`dtypes.uint8`、`dtypes.uint16`、`dtypes.uint32`、`dtypes.uint64`<br>布尔类型：`dtypes.bool`；`dtypes.bool` 是同一类型的别名 |
| `"nz"` | 浮点类型：`dtypes.float16`、`dtypes.bfloat16`、`dtypes.float32`、`dtypes.float8_e4m3fn`、`dtypes.float8_e5m2`、`dtypes.fp4x2_e2m1`<br>有符号整数：`dtypes.int8`<br>无符号整数：`dtypes.uint8` |

## 返回值说明

返回一个描述Tensor参数的 `TensorSpec` 对象。

## 约束说明

当 `storage_format="nz"` 时，还须满足以下要求：

| 项目 | 要求 |
| --- | --- |
| 形状 | 维数必须为2或3；静态长度必须大于0，直接使用的 `Dim` 必须满足 `min >= 1`。 |
| 步长 | 必须为根据 `shape` 推导出的紧凑行主序步长；显式指定时也须与推导结果一致。 |
| FP4类型 | 当 `dtype` 为 `dtypes.fp4x2_e2m1` 时，所有维度必须是静态整数，且最后一维必须为偶数。 |

## 调用示例

以下示例使用TensorSpec提前编译长度为8的Tensor复制函数，并校验输出与输入一致。

```python
import cannbotdsl as cbd
import torch
import torch_npu


@cbd.kernel
def copy_kernel(x: cbd.Tensor, y: cbd.Tensor):
    for i in range(x.shape[0]):
        y[i] = x[i]


@cbd.host
def copy(x: cbd.Tensor, y: cbd.Tensor):
    copy_kernel[1](x, y)


def main():
    spec = cbd.TensorSpec((8,), cbd.dtypes.float32)
    with cbd.compile(copy, spec, spec) as program:
        x = torch.arange(8, dtype=torch.float32, device="npu")
        y = torch.zeros_like(x)
        program(x, y)
        torch.npu.synchronize()
        torch.testing.assert_close(y.cpu(), x.cpu())
        print("output:", y.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
output: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
```
