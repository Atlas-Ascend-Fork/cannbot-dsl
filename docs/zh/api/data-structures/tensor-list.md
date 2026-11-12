# TensorList

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 简介

`TensorList`表示运行时传入的一组Tensor，可用于向CANNBot-DSL程序传递多个同类型、同维数的Tensor。

## 定义

在Host调用入口中使用`TensorList`类型注解，并传入非空Tensor列表或元组。框架根据传入的列表创建`TensorList`对象，无需直接调用构造函数。提前编译时可使用[TensorListSpec](../compilation/host/specs/tensor_list_spec.md)描述元素和列表长度。

| 属性或用法 | 说明 |
| --- | --- |
| `xs.length` | 获取列表长度。 |
| `xs[i]` | 在设备代码中按非负整数下标获取Tensor。 |

## 约束说明

- 列表不能为空，各Tensor的数据类型和维数须一致，且须满足入口参数的规格。
- 不支持直接遍历或使用`len(xs)`；在Kernel内可通过`xs.length`和下标访问元素。
- 不支持负数下标，运行时下标的有效范围由调用方保证。

## 使用示例

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
