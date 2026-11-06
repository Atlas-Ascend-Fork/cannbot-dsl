# dump_reg

## 产品支持情况

| 产品 | 支持情况 |
| --- | --- |
| Ascend 950PR&950DT系列产品 | 适用于dav-3510的SIMD VF。 |
| 其他产品 | 未声明支持。 |

## 功能说明

将一个矢量寄存器中的数据写入CANN调试流，输出寄存器位置标记、数据类型、`desc`及元素内容，用于V核SIMD VF调测。打印后保留源寄存器，可以继续参与计算。

## 函数原型

```python
from cannbotdsl import dump_reg

dump_reg(
    register,
    *,
    desc=None,
    dump_size=None,
    dtype=None,
) -> None
```

除`register`外，参数均须以关键字传入。

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| `register` | 输入 | SIMD VF中的`RawVReg`。不接受Tensor、普通标量、Buffer、RingBuffer或mask/谓词寄存器。 |
| `desc` | 输入 | 自定义uint32标识，范围为0～2³²−1。支持编译期整数和运行时整数值。默认按构图顺序给当前Kernel的dump操作编号，与旧print采集编号独立；同一运行时循环操作复用编号。编号不是跨核或跨线程的全局唯一标识。 |
| `dump_size` | 输入 | 元素数量，默认打印一个完整寄存器。支持编译期整数和运行时整数值。合法uint32请求大于容量时自动截断到容量。 |
| `dtype` | 输入 | 可选的`cannbotdsl.dtypes`显示类型，默认取寄存器元素类型。可独立于载体类型解释寄存器原始位，不做数值转换，打印容量随显示类型计算。 |

## 数据类型

一个寄存器的可打印载荷为256字节。容量由显示类型决定。

| 数据类型 | 单寄存器最多打印的元素数 | 支持情况 |
| --- | ---: | --- |
| `int8`、`uint8` | 256 | 支持。 |
| `bool` | 256 | 支持按一字节解释的数据；不是mask寄存器打印。 |
| `float8_e4m3fn`、`float8_e5m2`、`float8_e8m0`、`hifloat8` | 256 | 支持。 |
| `int16`、`uint16`、`float16`、`bfloat16` | 128 | 支持。 |
| `int32`、`uint32`、`float32` | 64 | 支持。 |
| `int64`、`uint64` | 32 | 支持。 |
| `fp4x2_e2m1`、`fp4x2_e1m2` | — | 当前不支持 FP4 可读数值输出。 |
| `float64`、int4及其他未列出的亚字节类型 | — | 不支持。 |

### FP4转换后打印

FP4数据可先做数值转换，再使用`dump_reg`打印转换后的矢量寄存器。示例使用E2M1 → BF16 → FP32路径，只处理一个分组，输出的是转换后的FP32数据。仅修改dtype或view不会完成数值转换。

`rr`为`cannbotdsl.ops.reg`；`vf`、`dtypes`和本页调测接口从`cannbotdsl`导入。代码位于Kernel内，省略在同一SIMD VF内将数据加载为E2M1寄存器`fp4`的语句。

```python
with vf(mode="simd"):
    bf16 = rr.vcast(fp4, dtypes.bfloat16,
                    mask=rr.create_mask("all", elem_bits=8), reg_layout=rr.RegLayout.ZERO)
    fp32 = rr.vcast(bf16, dtypes.float32,
                    mask=rr.create_mask("all", elem_bits=16), reg_layout=rr.RegLayout.ZERO)
    dump_reg(fp32, dump_size=8)
```

### 动态整数参数

API的uint32范围不自动扩大调用入口的参数位宽。运行时参数应显式选取足够宽的类型；例如需要完整覆盖0～2³²−1时，可将Host/Kernel参数均声明为ScalarValue，并在编译规格中使用`dtypes.uint32`或`dtypes.int64`。若参数实际为有符号int32，则不能传入大于2³¹−1的值来代替uint32。非法静态参数通常在构图或翻译时报错；非法动态desc/count被运行时保护条件跳过时，不会自动给出Python异常或警告。

## 返回值说明

返回`None`。源寄存器保持可用；文本由CANN宿主运行库异步输出。

## 约束说明

### 执行域与输入

- 只能在`vf(mode="simd")`中调用，不支持普通Kernel标量上下文、Host、SIMT VF或纯SIMT。
- 只能读取当前矢量寄存器。不能直接读取GM、UB、L1、L0A、L0B、L0C、BIAS、FIXBUF或SSBUF；需要先通过合法的数据搬运和加载取得寄存器，或使用支持相应位置的[dump_tensor](./dump_tensor.md)。
- `dtype`是位模式解释方式。例如字节寄存器可以按bool显示；该方式不提供类型转换，也不增加mask寄存器支持。

### 长度与缓冲

- 默认打印全部256字节。显式长度较小时只打印前部元素；大于容量的合法uint32请求自动截断，与AscendC寄存器接口一致。
- 负数或不能表示为uint32的count、desc，静态参数报错，动态参数跳过本次dump。运行时边界检查不会将非法整数先截断为uint32。
- 零长度不保证产生完整文本记录，不应依赖其打印效果。
- 与`dump_tensor`、Kernel `print`和原生断言共享2 KiB VF调测空间。单条寄存器载荷至多256字节，但多次调用仍受共享缓冲及GM日志容量限制。如果其他调测调用使本次VF缓冲进入错误状态，例如一条超限printf记录，则后续dump_reg也会被跳过；下一次重新初始化的VF调用可恢复。
- 具备`readLen/finish`协议的SDK可在同一次VF调用中排空、复用空间，累计输出可以超过2 KiB；旧`pidx`协议保留原有批次限制。不保证任意次数打印均可完整显示。
- 含VF dump的Kernel预留8 KiB UB；同时使用SIMD VF格式化print时合计预留10 KiB。手工存储地址也须避开保留区域。

### 输出与构建

- 输出进入CANN调试流，不进入`get_debug_prints()`。设备同步完成后，宿主日志刷新仍可能稍晚；调测会影响资源占用和执行耗时。
- 仅使用dump支持编译及AOT保存、加载；与旧采集式print混用时，Provider保存会被拒绝，具体见[print](./print.md)。
- `ASCENDC_DUMP`是编译期宏，使用`-DASCENDC_DUMP=0`编译并重新生成产物可关闭设备输出；仅设置同名Shell环境变量不生效，也不会改变已有产物。

## 相关接口

- [dump_tensor](./dump_tensor.md)：打印支持位置上的连续物理存储区间。
- [print](./print.md)：打印格式化标量、字符串和指针等。
