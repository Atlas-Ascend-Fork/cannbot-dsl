# 存储位置

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

`cannbotdsl.MemLoc` 是存储位置，用于指定数据所在或所需的存储层级，可作为相关接口的 `mem_loc` 参数。

当前共有 **12 个枚举成员**。

## 原型定义

| 枚举成员 | 含义 |
| --- | --- |
| `cannbotdsl.MemLoc.GM` | 全局内存，用于存放输入、输出及全局数据。 |
| `cannbotdsl.MemLoc.UB` | Unified Buffer，片上向量计算缓冲区。 |
| `cannbotdsl.MemLoc.L1` | L1 片上缓冲区，用于缓存计算数据。 |
| `cannbotdsl.MemLoc.L0A` | 矩阵计算的 A 操作数缓冲区。 |
| `cannbotdsl.MemLoc.L0B` | 矩阵计算的 B 操作数缓冲区。 |
| `cannbotdsl.MemLoc.L0C` | 矩阵计算的累加结果缓冲区。 |
| `cannbotdsl.MemLoc.RegFile` | 寄存器存储空间。 |
| `cannbotdsl.MemLoc.L0A_MX` | MX 矩阵计算中 A 操作数的缩放因子存储空间。 |
| `cannbotdsl.MemLoc.L0B_MX` | MX 矩阵计算中 B 操作数的缩放因子存储空间。 |
| `cannbotdsl.MemLoc.BIAS` | 矩阵计算的偏置缓冲区。 |
| `cannbotdsl.MemLoc.SSBUF` | SSBUF 片上存储空间。 |
| `cannbotdsl.MemLoc.FBUF` | FBUF 片上存储空间。 |
