# 片上存储空间

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

`cannbotdsl.XX.view()` 用于返回对应片上存储空间的 uint8 Tensor 根视图。可通过根视图的 [`reinterpret()`](../tensor/reinterpret.md) 方法派生子视图。满足接口约束的根视图和子视图可作为 [`make_buffer()`](buffer/make_buffer.md) 的 `view` 参数，或作为 [`make_ring_buffer()`](ring-buffer/make_ring_buffer.md) 的 `views` 参数中的元素。

## 原型定义

| 根视图 | 含义 |
| --- | --- |
| `cannbotdsl.UB.view(memloc_size)` | 返回覆盖完整 UB 空间的 uint8 Tensor 根视图。 |
| `cannbotdsl.L1.view(memloc_size)` | 返回覆盖完整 L1 空间的 uint8 Tensor 根视图。 |
| `cannbotdsl.L0A.view(memloc_size)` | 返回覆盖完整 L0A 空间的 uint8 Tensor 根视图。 |
| `cannbotdsl.L0B.view(memloc_size)` | 返回覆盖完整 L0B 空间的 uint8 Tensor 根视图。 |
| `cannbotdsl.L0C.view(memloc_size)` | 返回覆盖完整 L0C 空间的 uint8 Tensor 根视图。 |
| `cannbotdsl.BIAS.view(memloc_size)` | 返回覆盖完整 BIAS 空间的 uint8 Tensor 根视图。 |
| `cannbotdsl.SSBUF.view(memloc_size)` | 返回覆盖完整 SSBUF MIX 地址空间的 uint8 Tensor 根视图。 |
| `cannbotdsl.FBUF.view(memloc_size)` | 返回覆盖完整 FBUF 空间的 uint8 Tensor 根视图。 |

`memloc_size` 表示对应存储空间的完整字节容量。

## 约束说明

- 使用某个存储层级的根视图后，不能再通过 `Buffer()` 或 `RingBuffer()` 在该层级分配存储。
- 同一 Kernel 中，使用任一存储层级的根视图后，不能调用 [`cannbotdsl.onchip_rewind()`](../../execution/synchronization/onchip_rewind.md)。
- SSBUF 的完整容量表示 MIX 地址空间；各核只能访问自身允许的地址范围。
