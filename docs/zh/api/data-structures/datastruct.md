# @datastruct

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

`@datastruct`用于声明与C结构体兼容的数据类型。通过字段名称和类型组织参数，可将结构体对象传递给CANNBot-DSL程序。

## 定义与创建结构体

在类定义前添加`@cbd.datastruct`，并为每个字段声明类型。被装饰的类至少包含一个字段，不能继承其他类，字段不能设置默认值。

以下示例定义并创建包含两个字段的结构体。

```python
import cannbotdsl as cbd


@cbd.datastruct
class TileConfig:
    rows: cbd.dtypes.int32
    columns: cbd.dtypes.int32


config = TileConfig(rows=64, columns=128)
assert config.rows == 64
assert config.columns == 128
```

## 指定结构体名称或字段规格

- 使用`@cbd.datastruct(c_name="TileConfig")`指定对应C结构体的名称。
- 使用`@cbd.datastruct(schema=...)`指定[字段规格](../compilation/host/specs/struct_schema.md)。

`c_name`和`schema`不能同时指定。
