# 原子操作

数据搬运随路原子操作接口用于对后续目的地址为GM的数据搬运指令开启原子操作，涉及的接口请参见表1。如下图1的左侧子图所示，未开启原子操作时，写入GM的数据搬运完成后，GM中原始数据将被新搬运数据完全覆盖。如图1右侧子图所示，当数据搬运随路原子操作接口被调用后，系统将为后续写入GM的数据搬运开启原子操作。此时，数据搬运完成后，GM中的最终数据由原始GM数据与新搬运数据共同决定。

## 数据搬运随路原子操作

**图1** 数据搬运随路原子累加效果

![atomic operation pipelined data movement effect示意图](../../figures/atomic_operation_pipelined_data_movement_effect.svg)

**表1** 数据搬运随路原子操作接口

| 对应接口 | 接口功能描述 |
| --- | --- |
| [`mem_copy`](../../data-movement/mem-copy.md)（`options=copy_options(atomic_add=True)`） | 对本次目的地址为GM的数据搬运开启原子累加。原子累加过程：将待拷贝的内容和GM已有内容进行求和，然后将求和结果写入GM。 |

DSL目前只提供随路原子累加，未提供随路原子取大、取小以及关闭随路原子操作的接口；随路原子累加仅支持UB→GM与L0C→GM方向的搬运。

## 标量原子操作

Scalar原子操作接口用于对指定GM地址上的单个元素执行原子计算，涉及的接口请参见表2。它们在AIV的标量流水上执行，返回更新前的元素值，不会影响后续向GM搬运数据的指令。在Mix模式中，两个AIV子核都会执行未加保护的调用；要求每个逻辑block只更新一次时，须用`get_subblock_id() == 0`限制执行子核。

如下图2左侧子图所示，不使用`atomic_add`接口时，多个AI Core同时对同一GM地址执行累加操作会相互覆盖，操作不具备原子性，最终结果不可预期。如右侧子图所示，使用`atomic_add`接口后，各AI Core的累加操作串行化执行，确保每次累加操作的原子性，最终结果符合预期。

**图2** 标量原子累加效果

![scalar atomic operation diagram示意图](../../figures/scalar_atomic_operation_diagram.svg)

**表2** Scalar原子操作接口

| 对应接口 | 接口功能描述 |
| --- | --- |
| [`atomic_add`](./scalar_atomic/atomic-add.md) | 用于在指定GM地址上进行原子加操作，将address指向的GM地址上的旧值（`old_value`）与输入标量值（value）求和，将和结果（`new_value`）写回GM地址，返回该地址修改前的值（`old_value`）。 |
| [`atomic_min`](./scalar_atomic/atomic-min.md) | 用于在指定GM地址上进行原子取最小值操作，将address指向的GM地址上的旧值（`old_value`）与输入标量值（value）做比较，将较小值（`new_value`）写回GM地址，返回该地址修改前的值（`old_value`）。 |
| [`atomic_max`](./scalar_atomic/atomic-max.md) | 用于在指定GM地址上进行原子取大操作，将address指向的GM地址上的旧值（`old_value`）与输入的标量值（value）进行比较，将较大值（`new_value`）写回GM地址，返回该地址修改前的值（`old_value`）。 |
| [`atomic_cas`](./scalar_atomic/atomic-cas.md) | 在指定GM地址上进行原子比较操作，读取address指向的GM地址上的旧值（`old_value`）与输入标量值value1进行比较：如果相等，则将输入标量值value2写入GM地址；如果不相等，则GM地址上的值保持不变。返回该地址修改前的值（`old_value`）。 |
| [`atomic_exch`](./scalar_atomic/atomic-exch.md) | 用于在GM内存中执行原子交换操作，读取address指向的GM地址上的旧值（`old_value`），并将输入的标量值（value）替换旧值存储回同一地址，返回该地址修改前的值（`old_value`）。 |
| [`atomic_and`](./scalar_atomic/atomic-and.md) | 对GM中的数据与指定数据执行原子与操作，即将val按位与到address指向的数据元素上。读取address指向的GM地址上的旧值（`old_value`），将旧值与输入标量值val进行按位与运算，将结果（`new_value`）写回GM地址，返回该地址修改前的值（`old_value`）。 |
| [`atomic_inc`](./scalar_atomic/atomic-inc.md) | 对GM中address指向的计数器执行原子递增操作，如果address上的数值大于等于指定数值val，则对address赋值为0，否则将address上数值加1，返回该地址修改前的值（`old_value`）。 |
| [`atomic_dec`](./scalar_atomic/atomic-dec.md) | 对GM中address指向的计数器执行原子递减操作，如果address上的数值等于0或大于指定数值val，则对address赋值为val，否则将address上数值减1，返回该地址修改前的值（`old_value`）。 |
| [`atomic_or`](./scalar_atomic/atomic-or.md) | 对GM中的数据与指定数据执行原子或操作，即将val按位或到address指向的数据元素上。读取address指向的GM地址上的旧值（`old_value`），将旧值与输入标量值val进行按位或运算，将结果（`new_value`）写回GM地址，返回该地址修改前的值（`old_value`）。 |
| [`atomic_sub`](./scalar_atomic/atomic-sub.md) | 对GM中的数据与指定数据执行原子减操作，即将val从address指向的数据元素上减去。读取address指向的GM地址上的旧值（`old_value`），将旧值减去输入标量值val，将结果（`new_value`）写回GM地址，返回该地址修改前的值（`old_value`）。 |
| [`atomic_xor`](./scalar_atomic/atomic-xor.md) | 对GM中的数据与指定数据执行原子异或操作，即将val按位异或到address指向的数据元素上。读取address指向的GM地址上的旧值（`old_value`），将旧值与输入标量值val进行按位异或运算，将结果（`new_value`）写回GM地址，返回该地址修改前的值（`old_value`）。 |
