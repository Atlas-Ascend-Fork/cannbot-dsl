# 数据类型

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

`cannbotdsl.dtypes`用于指定元素的数据类型。


## 标量值

接口原型中的`ScalarValue`表示单个数值，即标量。例如，`x`为一维`int32` Tensor时，读取的`x[0]`就是一个`int32`标量。每个接口支持的数据类型见其参数说明。

## 类型列表

| 数据类型 | 说明 |
| --- | --- |
| `cannbotdsl.dtypes.float16` | 16 位浮点类型，采用 float16 格式。 |
| `cannbotdsl.dtypes.float32` | 32 位浮点类型，采用 float32 格式。 |
| `cannbotdsl.dtypes.float64` | 64 位浮点类型，采用 float64 格式。 |
| `cannbotdsl.dtypes.bfloat16` | 16 位浮点类型，采用 bfloat16 格式。 |
| `cannbotdsl.dtypes.float8_e4m3fn` | 8 位浮点类型，采用 float8_e4m3fn 格式。 |
| `cannbotdsl.dtypes.float8_e5m2` | 8 位浮点类型，采用 float8_e5m2 格式。 |
| `cannbotdsl.dtypes.float8_e8m0` | 8 位浮点类型，采用 float8_e8m0 格式。 |
| `cannbotdsl.dtypes.hifloat8` | 8 位浮点类型，采用 hifloat8 格式。 |
| `cannbotdsl.dtypes.int8` | 8 位有符号整数类型。 |
| `cannbotdsl.dtypes.int16` | 16 位有符号整数类型。 |
| `cannbotdsl.dtypes.int32` | 32 位有符号整数类型。 |
| `cannbotdsl.dtypes.int64` | 64 位有符号整数类型。 |
| `cannbotdsl.dtypes.uint8` | 8 位无符号整数类型。 |
| `cannbotdsl.dtypes.uint16` | 16 位无符号整数类型。 |
| `cannbotdsl.dtypes.uint32` | 32 位无符号整数类型。 |
| `cannbotdsl.dtypes.uint64` | 64 位无符号整数类型。 |
| `cannbotdsl.dtypes.bool` | 布尔类型，表示真或假。 |
| `cannbotdsl.dtypes.fp4x2_e2m1` | 两个 e2m1 格式的 4 位浮点元素打包在一个字节中。 |
| `cannbotdsl.dtypes.fp4x2_e1m2` | 两个 e1m2 格式的 4 位浮点元素打包在一个字节中。 |
| `cannbotdsl.dtypes.int4x2` | 两个 4 位有符号整数打包在一个字节中。 |

SIMT短向量类型见[短向量数据类型](../simt/data-types.md)。
