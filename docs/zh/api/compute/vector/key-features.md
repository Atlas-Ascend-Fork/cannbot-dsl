# 关键特性说明

## 硬件循环（HardwareLoop）

Reg 矢量计算架构中，矢量函数（Vector Function，VF）用于组织矢量寄存器计算。VF中可以包含最多四层嵌套循环，每层循环中还可以包含多个串行循环。VF循环对控制结构的支持有限，仅支持for循环和条件判断，不支持switch、do-while和while-do等其他控制结构。当VF中的循环满足Hardware Loop编码规范会被编译器优化为Hardware Loop，提升整体的编码性能，否则它的循环逻辑会由迭代变量和条件判断语句构成Software Loop，无法开启VF循环优化。

### Hardware Loop编码规范

- Hardware Loop最多能支持的嵌套层数为4层；支持串行的Hardware Loop循环。
- 迭代变量类型<br>VF内所有Loop的迭代变量必须是uint16_t类型。
- 起始值与步长<br>循环起始值从0开始。<br>每次迭代的步长必须是递增1。
- 循环内不允许跳转指令，比如条件判断跳转，如if/else、三元运算符?:。<br>VF内if/else在Loop内会阻碍Hardware Loop的生成，编译器虽然会尽可能的做if/else消除优化，但是不做完全性保证。
- 一旦执行，循环计数/边界不允许被更改。
- 若要利用外层循环的计数作为循环边界，将外层循环计数器赋值给标量后作为内层循环边界。

## 指令双发优化

### 指令双发

指令双发指的是处理器在同一个时钟周期内，能够同时发射两条指令到执行单元进行处理。

这种机制可以在不改变程序逻辑的前提下，提升处理器在单位时间内的指令处理效率，是实现指令级并行的重要基础之一。

### 合理拆分VF循环

可将中间结果写回Unified Buffer（UB），拆分过长的Vector Function，减少数据依赖。指令缓存未命中会增加取指等待时间。循环内包含同步时，可通过拆分循环将同步移到循环外，减少同步次数。

### 手动控制循环拆分

如果循环内存在依赖关系过多的指令，指令无法并发执行，即无法启用双发特性。可以通过`cb.range`的`unroll`参数展开循环，提升指令双发能力，贴近硬件乱序执行的特性；减少指令因为寄存器资源未到位而产生的等待。

## VF融合

VF融合将满足条件的多个Vector Function合并为一个Vector Function，减少中间数据搬运和调用开销。编译器根据数据依赖、指令支持情况等条件自动决定是否融合，并保持原有计算结果和执行依赖关系。

在DSL中，融合仅作用于用户显式指定的VF作用域（`with vf(mode="simd"):`）：将可融合的相邻矢量操作放在同一个VF作用域内，中间结果可以不经过UB中转；将需要独立同步或独立流水的操作拆分到不同的VF作用域，可以避免不必要的保守同步。

## 数据搬运优化

Reg 矢量计算接口提供了下表所示多种搬运指令，合理选择搬运接口并利用接口能力进行优化。

| 场景 | 接口 | 描述 |
| :-- | :-- | :-- |
| 连续对齐搬入 | [`vload`](./load/vload.md) | 从32字节对齐的UB起始地址连续搬入到向量数据寄存器或掩码寄存器。 |
| 非连续对齐搬入 | [`vload_strided`](./load/vload-strided.md) | 从32字节对齐的UB起始地址非连续搬入多个DataBlock。 |
| 连续非对齐搬入 | [`vload_unalign`](./load/vload-unalign.md) | 从非32字节对齐的UB起始地址连续搬入到向量数据寄存器。 |
| 掩码寄存器搬入 | [`vmask_load`](./load/vmask-load.md) | 从UB连续搬入掩码寄存器。 |
| 离散搬入 | [`vgather`](./ub_gather/vgather.md) | 根据索引值将UB中的元素收集到向量数据寄存器。 |
| DataBlock离散搬入 | [`vgather_datablock`](./ub_gather/vgather-datablock.md) | 根据索引值将UB中的元素按DataBlock（32B）收集到向量数据寄存器。 |
| 连续对齐搬出 | [`vstore`](./store/vstore.md) | 将向量数据寄存器或掩码寄存器连续搬出到32字节对齐的UB地址。 |
| 连续非对齐搬出 | [`vstore_unalign_post`](./store/vstore-unalign-post.md) | 将向量数据寄存器连续搬出到非32字节对齐的UB地址。 |
| 离散搬出 | [`vscatter`](./scatter/vscatter.md) | 根据索引值将向量数据寄存器中的元素离散搬出到UB。 |
| 寄存器间搬运 | [`vmask`](./reg_logic/vmask.md) | 将向量数据寄存器中被掩码筛选的有效元素复制到另一个向量数据寄存器。 |

### 数据搬运启用分布模式

- 连续对齐搬入

  如图1，将UB地址`x_addr`上数据量为VL/2的`dtypes.float16`元素通过[`vload_unpack`](./load/vload-unpack.md)搬入`x_reg`，调用[`vcast`](./reg_convert/vcast.md)（`reg_layout=cb.reg.RegLayout.ZERO`）将`dtypes.float16`转换为`dtypes.int32`并写入`y_reg`，最后通过[`vstore`](./store/vstore.md)搬出至UB地址`y_addr`。

  **图1** half转int32类型转换过程

  ![half to int32 conversion示意图](../../figures/half_to_int32_conversion.svg)

- 连续对齐搬出

  如图2，将UB地址`x_addr`上数据量为VL的`dtypes.float32`元素通过[`vload`](./load/vload.md)搬入`x_reg`，调用[`vcast`](./reg_convert/vcast.md)（`reg_layout=cb.reg.RegLayout.ZERO`）将`dtypes.float32`转换为`dtypes.int16`并写入`y_reg`，最后通过[`vstore_pack`](./store/vstore-pack.md)搬出至UB地址`y_addr`。

  **图2** float转int16类型转换过程

  ![float to int16 conversion示意图](../../figures/float_to_int16_conversion.svg)
