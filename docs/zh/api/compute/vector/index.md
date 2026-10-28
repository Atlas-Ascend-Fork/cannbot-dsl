# Reg 矢量计算

Reg 矢量计算接口在 `simd` 模式下操作矢量数据寄存器与掩码寄存器，均须在 VF 作用域内调用，仅在 AIV 上生效。

- **[概述](./overview.md)**
- **[关键特性说明](./key-features.md)**

## 子分类

- **[Reg 数据搬入](./load/overview.md)**
- **[基础算术](./reg_arith/vabs.md)**
- **[广播操作](./reg_broadcast/vdup.md)**
- **[比较计算](./reg_compare/veq.md)**
- **[类型转换](./reg_convert/roundingmode.md)**
- **[复合计算](./reg_fused/vabs-sub.md)**
- **[聚合操作](./reg_gather/vgather-reg.md)**
- **[直方图](./reg_histogram/vhistogram-accumulate.md)**
- **[索引操作](./reg_index/varange.md)**
- **[逻辑计算](./reg_logic/mask-and.md)**
- **[掩码寄存器操作](./reg_mask/create-mask.md)**
- **[排布变换](./reg_permute_sel/vcompress.md)**
- **[归约计算](./reg_reduce/vpair-reduce-sum.md)**
- **[同步控制](./reg_sync/vmem-bar.md)**
- **[Reg 离散搬出](./scatter/vscatter.md)**
- **[Reg 数据搬出](./store/overview.md)**
- **[Reg 聚合搬入](./ub_gather/vgather-datablock.md)**
