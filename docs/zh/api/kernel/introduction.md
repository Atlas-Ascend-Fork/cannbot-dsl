# Kernel API


Kernel API 面向 AI Core，在 `@kernel` 函数体或其展开的设备侧辅助函数中使用，用于描述设备上的数据访问、计算、同步和控制逻辑。

有关 `@host`、`@jit` 和 `@kernel` 的定义及基本调用方式，请参见[装饰器](/zh/api/decorators)。

## 接口分类

| 分类 | 主要内容 | 接口 |
| --- | --- | --- |
| 基本数据类型与操作 | dtype、Tensor、shape、layout、坐标、构造和视图 | `DType`、`Tensor`、`Shape`、`Layout`、`make_tensor`、`tile_view` |
| 数据搬运 | 存储层级间搬运和搬运引擎配置 | [`mem_copy`](/zh/api/kernel/data-movement/mem-copy)、[`make_copy_engine`](/zh/api/kernel/data-movement/make-copy-engine) |
| [Reg矢量计算](/zh/api/kernel/reg_compute/) | `simd` 模式下矢量数据寄存器与掩码寄存器的算术、复合计算、比较、归约、逻辑、类型转换、排布变换、广播、索引、直方图、掩码寄存器操作，以及 Reg数据搬入、Reg聚合搬入、Reg数据搬出、Reg离散搬出 | `vadd`、`vselect`、`vreduce_sum`、`vcast`、`vload`、`vstore`、`create_mask` 等 |
| [标量计算](/zh/api/kernel/scalar_compute/) | 标量流水（`PIPE_S`）上的位运算、类型转换与不经过DCache的GM访存 | `popc`、`clz`、`ffs`、`cast`、`load_bypass` 等 |
| [原子操作](/zh/api/kernel/atomic/) | GM 地址上的单点原子计算操作，多个 AI Core 的操作串行化执行（绕过 DCache） | `atomic_add`、`atomic_cas`、`atomic_inc` 等 |
| [矩阵计算](/zh/api/kernel/cube_compute/) | Mmad 矩阵乘加与 HF32/HiF8/FP8 计算模式配置 | `matmul`、`enable_hf32`、`set_mmad_direction` 等 |
| [同步与缓存控制](/zh/api/kernel/synchronization-cache/) | Cube、Vector、全局同步和缓存维护 | [`dcci_single`](/zh/api/kernel/synchronization-cache/dcci-single)、[`dcci_entire_out`](/zh/api/kernel/synchronization-cache/dcci-entire-out)、[`dcci_entire_atomic`](/zh/api/kernel/synchronization-cache/dcci-entire-atomic)、[`dci`](/zh/api/kernel/synchronization-cache/dci) |
| [同步管理](/zh/api/kernel/resource-management/) | 核间同步 | [核间同步接口](/zh/api/kernel/resource-management/)（[`cube_sync_block_arrive`](/zh/api/kernel/resource-management/cube-sync-block-arrive)、[`global_sync_all`](/zh/api/kernel/resource-management/global-sync-all) 等 9 个） |
| [系统变量访问](/zh/api/kernel/system/) | block、物理核、子核、周期和状态查询 | [`get_block_idx`](/zh/api/kernel/system/get-block-idx)、[`get_block_num`](/zh/api/kernel/system/get-block-num)、[`get_core_id`](/zh/api/kernel/system/get-core-id) 等 |
| 资源管理 | 统一缓冲区（UB）、Buffer、Channel 和延迟线 | `UB`、`make_buffer`、`make_channel`、`DelayLine` |
| 分布式通信 | 通信上下文、通信引擎和设备地址 | `distributed.get_comm_context`、`distributed.CommEngine` 等 |
| 调试接口 | 设备侧标量、Tensor 和寄存器调试 | [`print`](/zh/api/kernel/debug/print)、`dump_reg`、`dump_tensor` |
| 作用域与控制流 | 设备控制流、编译期表达式和执行作用域 | `range`、`select`、`const_expr`、`range_constexpr`、`cube`、`vf` |
