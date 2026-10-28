# TensorListSpec
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

`TensorListSpec` 用于描述Tensor列表的元素规格和列表长度，供 `cannbotdsl.compile(...)` 提前编译使用。

## 函数原型

```python
TensorListSpec(
    element: TensorSpec,
    length: int | Dim,
) -> TensorListSpec
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `element` | 输入 | 列表中各Tensor应满足的规格，使用 [TensorSpec](./tensor_spec.md) 描述。 |
| `length` | 输入 | 列表中的Tensor数量。固定数量使用整数，取值范围为 `[1, 2**63 - 1]`；动态数量使用 [Dim](./dim.md)。 |

## 返回值说明

返回一个描述Tensor列表参数的 `TensorListSpec` 对象。

## 约束说明

- `element` 的 `storage_format` 必须为 `"nd"`。
- `length` 使用 `Dim` 时，其 `min` 必须大于或等于 `1`。
- `@host` 函数中接收该列表的参数必须标注为 `TensorList`。

## 调用示例

以下示例使用TensorListSpec编译列表参数，将第一个Tensor复制到输出并校验结果。

```python
import cannbotdsl as cbd
import torch
import torch_npu


@cbd.kernel
def copy_first_kernel(xs: cbd.TensorList, y: cbd.Tensor):
    first = xs[0]
    for i in range(first.shape[0]):
        y[i] = first[i]


@cbd.host
def copy_first(xs: cbd.TensorList, y: cbd.Tensor):
    copy_first_kernel[1](xs, y)


def main():
    spec = cbd.TensorSpec((8,), cbd.dtypes.float16)
    list_spec = cbd.TensorListSpec(spec, length=2)
    with cbd.compile(copy_first, list_spec, spec) as program:
        xs = [
            torch.arange(8, dtype=torch.float16, device="npu"),
            torch.full((8,), 10, dtype=torch.float16, device="npu"),
        ]
        y = torch.zeros_like(xs[0])
        program(xs, y)
        torch.npu.synchronize()
        torch.testing.assert_close(y.cpu(), xs[0].cpu())
        print("output:", y.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
output: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
```
