# 短向量数据类型

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 简介

短向量由同一数据类型的多个分量组成，可用于SIMT接口。可通过[make_short_vector](./conversion/make-short-vector.md)构造短向量。

## 定义

| 数据类型 | 说明 |
| --- | --- |
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
