# dump_tensor

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将Tensor视图中连续的物理存储区间写入CANN调试流，由CANN宿主运行库输出数据类型、存储位置、地址、`desc`和元素内容。支持GM、UB、L1、L0C、BIAS、FBUF和FBUF_ACT的运行时调测，各存储位置的执行域和HDK版本要求见[存储位置与执行域](#存储位置与执行域)。须在Kernel或继承其设备域的jit中调用；普通Python和Host上下文不支持直接调用。

## 函数原型

```python
from cannbotdsl import dump_tensor

dump_tensor(
    tensor,
    *,
    desc=None,
    dump_size=None,
    offset=0,
    shape=None,
    dtype=None,
) -> None
```

除`tensor`外，参数均须以关键字传入。

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| `tensor` | 输入 | DSL Tensor视图。存储位置须在下表支持范围内。RingBuffer应先选取槽位并取得Tensor视图；不直接接受整个RingBuffer或裸Storage。 |
| `desc` | 输入 | 自定义uint32标识，范围为0～2³²−1。支持编译期整数和运行时整数值。默认按编译时遇到调用的顺序，为当前Kernel的转储操作编号，与旧print采集编号独立；同一运行时循环操作复用编号。编号不是跨核或跨线程的全局唯一标识。 |
| `dump_size` | 输入 | 元素数量，支持编译期整数和运行时整数值。默认取静态物理范围扣除`offset`后的剩余部分，最多1 KiB。显式指定可以超过1 KiB；数量及数据字节数均须能用uint32表示。 |
| `offset` | 输入 | 相对于当前视图起点的非负元素偏移，默认为0。支持编译期整数和运行时整数值，不接受bool或浮点数。 |
| `shape` | 输入 | 可选的编译期tuple或list，最多8维，每一维为正uint32整数。默认不附加显示维度。只控制显示元数据，不改变实际读取顺序及`dump_size`；VF内不接受非空显示维度。 |
| `dtype` | 输入 | 可选的`cannbotdsl.dtypes`显示类型，默认使用Tensor元素类型。只能在保持元素存储位宽的条件下重新解释数据，不进行数值转换。bool占一字节。 |

### 存储位置与执行域

| 输入位置 | 普通Kernel标量上下文 | SIMD VF | SIMT |
| --- | --- | --- | --- |
| GM | 支持，在当前Kernel对应核执行 | 不支持直接读取 | 本接口不支持 |
| UB | 支持，在AIV执行 | 支持 | 本接口不支持 |
| L1 | 支持，在AIC执行；须满足下文HDK版本要求 | 不支持 | 不支持 |
| L0C | 支持，在AIC执行 | 不支持 | 不支持 |
| BIAS | 支持，在AIC执行；须满足下文HDK版本要求 | 不支持 | 不支持 |
| FBUF、FBUF_ACT | 支持，在AIC执行；CANN输出位置为FIXBUF，须满足下文HDK版本要求 | 不支持 | 不支持 |
| L0A、L0B、SSBUF及其他位置 | 不支持 | 不支持 | 不支持 |

FIXBUF与SSBUF是不同存储区域。FBUF视图必须指向已有、有效的存储及正确的地址选择位；dump不会分配、初始化或修正该存储。

## 数据类型

| 类型 | DSL类型 | 支持范围 |
| --- | --- | --- |
| 有符号整数 | `int8`、`int16`、`int32`、`int64` | 支持。 |
| 无符号整数 | `uint8`、`uint16`、`uint32`、`uint64` | 支持。 |
| 常规浮点 | `float16`、`bfloat16`、`float32` | 支持。 |
| FP8及HiFloat8 | `float8_e4m3fn`、`float8_e5m2`、`float8_e8m0`、`hifloat8` | 支持。 |
| 布尔 | `bool` | 按一字节布尔数据解释。可用同位宽字节Tensor指定`dtype=bool`。 |
| FP4 | `fp4x2_e2m1`、`fp4x2_e1m2` | 当前不支持 FP4 可读数值输出。 |
| FP64、int4及其他未列出的亚字节类型 | — | 不支持。 |

使用时须同时满足数据类型、存储位置和执行域要求。

### FP4转换后打印

FP4数据可先做数值转换并写入受支持类型的Tensor，再使用`dump_tensor`打印。示例使用E2M1 → BF16 → FP32路径，只处理一个分组，输出的是转换后的FP32数据。仅修改dtype或view不会完成数值转换。

`rr`为`cannbotdsl.ops.reg`；`vf`、`dtypes`和本页调测接口从`cannbotdsl`导入。代码位于Kernel内，省略在同一SIMD VF内将数据加载为E2M1寄存器`fp4`的语句。 `decoded`是已分配的FP32 UB Tensor，至少容纳64个元素。

```python
with vf(mode="simd"):
    bf16 = rr.vcast(fp4, dtypes.bfloat16,
                    mask=rr.create_mask("all", elem_bits=8), reg_layout=rr.RegLayout.ZERO)
    fp32 = rr.vcast(bf16, dtypes.float32,
                    mask=rr.create_mask("all", elem_bits=16), reg_layout=rr.RegLayout.ZERO)
    rr.vstore(decoded, 0, fp32, rr.create_mask("all", elem_bits=32))
    rr.vmem_bar("vst_vld")

dump_tensor(decoded, dump_size=8)
```

### 动态整数参数

API的uint32范围不自动扩大调用入口的参数位宽。运行时参数应显式选取足够宽的类型；例如，要覆盖0～2³²−1的完整范围，参数应使用`dtypes.uint32`或能够容纳该范围的`dtypes.int64`，并与编译时指定的参数类型一致。若参数实际为有符号int32，则不能传入大于2³¹−1的值来代替uint32。以整数常量传入的`desc`或`dump_size`超出uint32范围时，接口在编译期报错；运行时计算得到的`desc`或`dump_size`超出该范围时，跳过本次转储，不抛出Python异常或警告。

## 返回值说明

返回`None`。输出进入CANN调试流；设备执行完成与宿主日志刷新不是同一时刻，不能通过返回值取得打印内容。

## 约束说明

### 长度、布局和地址

- 默认1 KiB是省略`dump_size`时的采样上限，不是普通Kernel上下文显式dump的固定上限。对应FP32最多256个、FP16最多512个、int8或bool最多1024个元素。
- 编译时已能确定的范围越界会在编译阶段报错。动态物理范围或动态偏移必须显式指定`dump_size`；调用者保证运行时地址、长度及向上取整后的底层读取范围有效。普通类型的动态offset和动态count不会自动按Tensor实际范围裁剪；越界可能读取错误数据或引发设备错误，不能把uint32范围校验当作内存边界校验。
- 零长度不保证产生完整记录：标量路径不输出数据记录，VF路径可能仅输出不完整的记录头。
- 输出按连续物理存储区间读取，不按逻辑stride收集数据。NZ/ZN物理顺序不保证等于逻辑矩阵顺序。L0C按硬件块搬出并在块内转换，不能将多块结果视为整个矩阵的行优先展开。
- 标量上下文的UB、L1源地址须按32字节对齐，L0C源地址须按1024字节对齐。部分长度也可能触发完整硬件块读取，调用者须提供有效尾部空间。
- `shape`与打印数量独立，不限制count，不进行reshape或gather；实际多维文本效果取决于CANN宿主解码器。

### SIMD VF与缓冲

- 仅允许`vf(mode="simd")`中的UB Tensor；进入VF前选好RingBuffer槽位，不支持SIMT VF、不支持VF内GM直接读取。
- 单条UB数据最多1952字节。编译时已能确定超限的请求会报错；动态请求超限时只跳过本次dump，不截断、不分片，后续合法记录仍可输出。当前SDK的2 KiB共享UB调测区扣除头部后有2024字节记录区，dump另占72字节管理信息，数据按32字节向上对齐。
- 当所用CANN版本支持排空并复用调试缓冲区时，可在同一次VF调用中重复利用缓冲空间，累计输出可以超过2 KiB；单条限制不变。不支持该能力时，仍受单批次容量限制。
- 与`dump_reg`、Kernel `print`及原生断言共享调测缓冲，仍受GM日志环形缓冲和宿主消费能力限制，不保证任意输出量均完整显示。如果其他调测调用使本次VF缓冲进入错误状态，例如一条超限printf记录，则后续dump也可能被跳过；下一次重新初始化的VF调用可恢复。
- 含VF dump的Kernel预留8 KiB UB，其中2 KiB用于调测、6 KiB用于VF栈；同时使用SIMD VF格式化print时再预留2 KiB。手工地址视图也须避开保留区。

### 大张量

- 当前不支持大张量分片的完整文本输出。需要查看大张量时，应拆为满足单条容量要求的小批量dump。VF单条数据仍限制为1952字节。

### 版本要求与输出

- 在Ascend 950系列产品上，打印L1、BIAS和FIXBUF数据要求HDK 25.7.RC1.6或更高版本。
- CANN文档提示每核dump总量需控制在30 KB范围内，并提供部分场景下的SIMD缓冲配置。具体容量及配置适用性取决于SDK和启动方式；本接口没有独立的无限输出缓冲。30 KB是日志预算，不等于可打印30 KB纯数据，还须扣除记录头和对齐开销；大张量应拆为小批量调用，避免依赖分片的完整文本输出。
- `ASCENDC_DUMP`是编译期宏，使用`-DASCENDC_DUMP=0`编译并重新生成产物可关闭设备调测输出；仅设置同名Shell环境变量不生效，也不会改变已有产物。仅调用CANN dump支持普通编译及AOT保存、加载；与旧采集式print混用时，保存仍会被拒绝，具体见[print](./print.md)。
- 输出异步；采集结果前应同步设备，宿主文本刷新仍可能稍晚。dump会影响同步、资源占用和性能，性能测量前应移除调测调用。

## 相关接口

- [dump_reg](./dump_reg.md)：打印SIMD VF中的矢量寄存器。
- [print](./print.md)：Python、Host和Kernel打印及旧采集模式。
