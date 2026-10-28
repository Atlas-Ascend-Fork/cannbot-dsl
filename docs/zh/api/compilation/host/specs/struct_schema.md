# StructSchema
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

定义结构体的名称、字段和字段类型，可用于创建 [StructSpec](./struct_spec.md)，或通过 `@datastruct(schema=...)` 定义对应的Python结构体类型。

## 函数原型

```python
StructSchema(name: str, fields: tuple[StructField, ...]) -> StructSchema
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `name` | 输入 | 结构体名称，须以英文字母开头，只包含字母、数字和下划线，不能使用 `class`、`if`、`return` 等语言关键字。 |
| `fields` | 输入 | 结构体的字段列表，每个字段使用 [StructField](#structfield) 定义。至少包含一个字段，字段名称不能重复，按传入顺序排列。 |

<a id="structfield"></a>

每个 `StructField` 定义一个字段的名称和类型。

```python
StructField(name: str, dtype: Any) -> StructField
```

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `name` | 输入 | 字段名称，须以英文字母开头，只包含字母、数字和下划线，不能使用 `class`、`if`、`return` 等语言关键字。 |
| `dtype` | 输入 | 字段保存的数据类型，可以是单个数值、固定长度数组（`CArray`）或结构体。支持的具体类型见 [支持的字段类型](#支持的字段类型)。 |

### 支持的字段类型

| 分类 | 支持的类型 |
| --- | --- |
| 浮点数 | `dtypes.float32`、`dtypes.float64` |
| 有符号整数 | `dtypes.int8`、`dtypes.int16`、`dtypes.int32`、`dtypes.int64` |
| 无符号整数 | `dtypes.uint8`、`dtypes.uint16`、`dtypes.uint32`、`dtypes.uint64` |
| 布尔值 | `dtypes.bool` |
| 固定长度数组 | 以上述标量类型为元素的 `CArray` |
| 结构体 | `StructSchema` 或由 `@datastruct` 定义的结构体类型 |

## 返回值说明

返回一个描述结构体名称、字段及字段类型的 `StructSchema` 对象。

## 约束说明

无。

## 调用示例

以下示例定义包含一个整数字段的结构体，并打印结构体名称和字段名称。完整的编译与运行流程见 [StructSpec](./struct_spec.md)。

```python
import cannbotdsl as cbd


def main():
    schema = cbd.StructSchema(
        "Options",
        (cbd.StructField("value", cbd.dtypes.int64),),
    )
    print("name:", schema.name)
    print("fields:", [field.name for field in schema.fields])


if __name__ == "__main__":
    main()
```

预期结果：

```text
name: Options
fields: ['value']
```
