# 数据类型

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

`cannbotdsl.dtypes` 模块提供只读 `cannbotdsl.DType` 对象，用于指定元素的数据类型，可直接作为支持该类型的接口的 `dtype` 参数。


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
| `cannbotdsl.dtypes.int8x2` | 由 2 个 int8 元素组成的短向量，总位宽为 16 位。 |
| `cannbotdsl.dtypes.int8x3` | 由 3 个 int8 元素组成的短向量，总位宽为 24 位。 |
| `cannbotdsl.dtypes.int8x4` | 由 4 个 int8 元素组成的短向量，总位宽为 32 位。 |
| `cannbotdsl.dtypes.uint8x2` | 由 2 个 uint8 元素组成的短向量，总位宽为 16 位。 |
| `cannbotdsl.dtypes.uint8x3` | 由 3 个 uint8 元素组成的短向量，总位宽为 24 位。 |
| `cannbotdsl.dtypes.uint8x4` | 由 4 个 uint8 元素组成的短向量，总位宽为 32 位。 |
| `cannbotdsl.dtypes.int16x2` | 由 2 个 int16 元素组成的短向量，总位宽为 32 位。 |
| `cannbotdsl.dtypes.int16x3` | 由 3 个 int16 元素组成的短向量，总位宽为 48 位。 |
| `cannbotdsl.dtypes.int16x4` | 由 4 个 int16 元素组成的短向量，总位宽为 64 位。 |
| `cannbotdsl.dtypes.uint16x2` | 由 2 个 uint16 元素组成的短向量，总位宽为 32 位。 |
| `cannbotdsl.dtypes.uint16x3` | 由 3 个 uint16 元素组成的短向量，总位宽为 48 位。 |
| `cannbotdsl.dtypes.uint16x4` | 由 4 个 uint16 元素组成的短向量，总位宽为 64 位。 |
| `cannbotdsl.dtypes.int32x2` | 由 2 个 int32 元素组成的短向量，总位宽为 64 位。 |
| `cannbotdsl.dtypes.int32x3` | 由 3 个 int32 元素组成的短向量，总位宽为 96 位。 |
| `cannbotdsl.dtypes.int32x4` | 由 4 个 int32 元素组成的短向量，总位宽为 128 位。 |
| `cannbotdsl.dtypes.uint32x2` | 由 2 个 uint32 元素组成的短向量，总位宽为 64 位。 |
| `cannbotdsl.dtypes.uint32x3` | 由 3 个 uint32 元素组成的短向量，总位宽为 96 位。 |
| `cannbotdsl.dtypes.uint32x4` | 由 4 个 uint32 元素组成的短向量，总位宽为 128 位。 |
| `cannbotdsl.dtypes.int64x2` | 由 2 个 int64 元素组成的短向量，总位宽为 128 位。 |
| `cannbotdsl.dtypes.int64x3` | 由 3 个 int64 元素组成的短向量，总位宽为 192 位。 |
| `cannbotdsl.dtypes.int64x4` | 由 4 个 int64 元素组成的短向量，总位宽为 256 位。 |
| `cannbotdsl.dtypes.uint64x2` | 由 2 个 uint64 元素组成的短向量，总位宽为 128 位。 |
| `cannbotdsl.dtypes.uint64x3` | 由 3 个 uint64 元素组成的短向量，总位宽为 192 位。 |
| `cannbotdsl.dtypes.uint64x4` | 由 4 个 uint64 元素组成的短向量，总位宽为 256 位。 |
| `cannbotdsl.dtypes.float32x2` | 由 2 个 float32 元素组成的短向量，总位宽为 64 位。 |
| `cannbotdsl.dtypes.float32x3` | 由 3 个 float32 元素组成的短向量，总位宽为 96 位。 |
| `cannbotdsl.dtypes.float32x4` | 由 4 个 float32 元素组成的短向量，总位宽为 128 位。 |
| `cannbotdsl.dtypes.float16x2` | 由 2 个 float16 元素组成的短向量，总位宽为 32 位。 |
| `cannbotdsl.dtypes.bfloat16x2` | 由 2 个 bfloat16 元素组成的短向量，总位宽为 32 位。 |
| `cannbotdsl.dtypes.fp8x2_e4m3fn` | 由 2 个 float8_e4m3fn 元素组成的短向量，总位宽为 16 位。 |
| `cannbotdsl.dtypes.fp8x2_e5m2` | 由 2 个 float8_e5m2 元素组成的短向量，总位宽为 16 位。 |
| `cannbotdsl.dtypes.hifloat8x2` | 由 2 个 hifloat8 元素组成的短向量，总位宽为 16 位。 |
