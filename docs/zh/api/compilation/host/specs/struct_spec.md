# StructSpec
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

`StructSpec` 用于描述结构体参数，供 `cannbotdsl.compile(...)` 提前编译使用。

## 函数原型

```python
StructSpec(schema: StructSchema) -> StructSpec
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `schema` | 输入 | 结构体的字段名称和类型定义，使用 [StructSchema](./struct_schema.md) 描述。 |

## 返回值说明

返回一个描述结构体参数的 `StructSpec` 对象。

## 约束说明

无。

## 调用示例

以下示例使用StructSpec提前编译结构体参数，将结构体字段写入Tensor并打印结果。

```python
import cannbotdsl as cbd
import torch
import torch_npu


OPTIONS_SCHEMA = cbd.StructSchema(
    "Options",
    (cbd.StructField("value", cbd.dtypes.int64),),
)


@cbd.datastruct(schema=OPTIONS_SCHEMA)
class Options:
    value: int


@cbd.kernel
def write_kernel(output: cbd.Tensor, options: Options):
    output[0] = options.value


@cbd.host
def run(output: cbd.Tensor, options: Options):
    write_kernel[1](output, options)


def main():
    output_spec = cbd.TensorSpec((1,), cbd.dtypes.int64)
    options_spec = cbd.StructSpec(OPTIONS_SCHEMA)
    with cbd.compile(run, output_spec, options_spec) as program:
        output = torch.zeros(1, dtype=torch.int64, device="npu")
        program(output, Options(value=7))
        torch.npu.synchronize()
        print("output:", output.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
output: [7]
```
