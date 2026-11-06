# print

## 产品支持情况

| 执行位置 | 支持情况 |
| --- | --- |
| Kernel、SIMD VF、SIMT | 适用于Ascend 950PR&950DT系列的dav-3510；其他产品未声明支持。 |
| 编译后的Host | 使用CPU运行库；HostLongDouble要求16字节原生ABI。 |
| 普通Python | 使用Python内置print，不依赖NPU执行。 |

## 功能说明

统一从`cannbotdsl`导入`print`，根据调用上下文选择输出方式，不需要通过别名区分Host和Kernel。DSL代码中的运行时打印须使用`from cannbotdsl import print`导入的接口，或通过`cannbotdsl.print`（包括模块别名形式）调用。Python内置print不是DSL运行时打印接口。

| 调用上下文／形式 | 执行行为 |
| --- | --- |
| 普通Python | 转发给Python内置print，保持其参数和格式语义。 |
| `@host`及Host域`@jit` | 生成CPU运行时格式化输出；每次执行到调用位置时打印当次参数。 |
| `@kernel`及Kernel域`@jit`，首参为字符串 | 使用CANN调试流执行Kernel格式化打印，对应AscendC printf。 |
| SIMD VF内格式化打印 | 在AIV的VF中执行，使用共享UB调测缓冲。 |
| SIMT VF或纯SIMT内格式化打印 | 每个执行到该位置的线程独立打印，不保证线程间显示顺序。 |
| Kernel内首参为Tensor或DSL标量 | 使用原有采集模式，保留结构化记录；不属于CANN printf输出。 |

构图、编译及AOT加载本身不会执行Host/Kernel格式化打印。Host使用C标准输出并刷新；Kernel由CANN异步处理，二者没有全局显示顺序保证。格式化print不会自动同步设备；原有采集模式则会执行Host回读和同步。

## 函数原型

公开入口按上下文分派：

```python
from cannbotdsl import print

print(*objects, **kwargs) -> None
```

Host和Kernel格式化调用形式：

```python
print(fmt, *args, end="\n") -> None
```

Kernel原有采集调用形式：

```python
print(value, *, label=None, mode="ring") -> None
```

格式化调用推荐通过位置传入首参，Host的fmt必须按位置传入。无参数调用、直接传Python数值或任意对象的内置print用法，仅适用于普通Python上下文；DSL内应使用格式字符串或受支持的采集值。

## 参数说明

### Host和Kernel格式化打印

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| `fmt` | 输入 | 编译期Python字符串。普通文本原样输出，转换说明与`args`逐项匹配，`%%`输出百分号。不能含NUL，不接受运行时格式字符串指针。 |
| `args` | 输入 | 与格式匹配的标量、字符串或类型化指针。支持多参数；类型、符号、位宽和参数数量须满足下文要求。Host动态宽度/精度的每个`*`也消耗一个参数。 |
| `end` | 输入 | 编译期字符串或`None`，默认`"\n"`。`None`同样追加换行；空字符串不追加。格式中已有换行时不会去重。不能含NUL；其中百分号按普通文本处理。 |

格式化调用不接受`label`、`mode`、`sep`、`file`、`flush`。普通Python调用保留内置print对这些名称的原有规则，其中`label/mode`并非内置参数。

### Kernel原有采集模式

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| `value` | 输入 | UB/L1/L0C Tensor，或Kernel局部及GM/UB加载得到的DSL标量。 |
| `label` | 输入 | 编译期字符串或`None`，默认自动生成记录标签。 |
| `mode` | 输入 | `"ring"`保留最后64条，`"stop"`保留最前64条；每个打印点在每个block/subblock上独立计数，停止保存不停止Kernel。 |

## 数据类型

### Kernel格式化打印

| 参数类型 | 支持的格式 | 说明 |
| --- | --- | --- |
| 8/16/32位有符号或无符号整数 | `%d`、`%i`、`%u`、`%x`、`%X` | 十进制符号须匹配；十六进制支持按整数位模式显示。较小位宽提升到格式所需宽度。 |
| 64位整数 | 上述整数格式的`l`或`ll`形式 | 必须匹配64位宽度，禁止隐式截断。 |
| `index` | `%ld`、`%lld`、`%li`、`%lli` | 按有符号64位处理；其他表示须先显式转换。 |
| FP16、BF16、FP32 | `%f`、`%F` | 以实际CANN文本输出为准，不套用Python的特殊值显示规则。 |
| 编译期字符串 | `%s` | 存储于对应地址空间，不能含NUL。 |
| 运行时零结尾8-bit字符指针 | `%s` | 普通Kernel和SIMT使用GM，SIMD VF使用UB。 |
| 类型化指针 | `%p` | 仅输出地址，不读取指向的数据；不接受普通整数冒充指针。 |

当前不支持 FP4 可读数值输出。

Python整数或浮点字面量作为`%f/%F`参数时，Kernel按FP32构图，可能发生舍入；动态整数值不会因此自动转为浮点，需先显式转换。整数格式按格式要求归一化为32/64位；有符号小整数用于十六进制时会先扩展，例如int8的−1使用`%x`显示为`ffffffff`。

Kernel不接受bool、FP64、FP8、Tensor本体、RawVReg或任意Python对象作为格式化标量。不支持宽度、精度、flags、`hh/h/j/z/t`整数长度、`%lf`、`%c/%o/%e/%E/%g/%G/%a/%A`。需要打印Tensor或寄存器时使用[dump_tensor](./dump_tensor.md)或[dump_reg](./dump_reg.md)，打印单个元素时先取得支持的标量值。

### FP4转换后打印

FP4数据可先做数值转换并写入受支持类型的Tensor，再读取标量，使用Kernel `print`打印。示例使用E2M1 → BF16 → FP32路径，只处理一个分组，输出的是转换后的FP32数据。仅修改dtype或view不会完成数值转换。

`rr`为`cannbotdsl.ops.reg`；`vf`、`dtypes`和本页调测接口从`cannbotdsl`导入。代码位于Kernel内，省略在同一SIMD VF内将数据加载为E2M1寄存器`fp4`的语句。 `decoded`是已分配的FP32 UB Tensor，至少容纳64个元素。

```python
with vf(mode="simd"):
    bf16 = rr.vcast(fp4, dtypes.bfloat16,
                    mask=rr.create_mask("all", elem_bits=8), reg_layout=rr.RegLayout.ZERO)
    fp32 = rr.vcast(bf16, dtypes.float32,
                    mask=rr.create_mask("all", elem_bits=16), reg_layout=rr.RegLayout.ZERO)
    rr.vstore(decoded, 0, fp32, rr.create_mask("all", elem_bits=32))
    rr.vmem_bar("vst_vld")

print("%f", decoded[(0,)])
```

### Host格式化打印

| 参数类型 | 支持范围 |
| --- | --- |
| 8/16/32/64位整数及index | `%d/%i/%u/%o/%x/%X`，支持`hh/h/l/ll/j/z/t`整数长度。参数先按格式要求提升为32/64位；`h/hh`随后按C规则窄化，不提供8/16位范围检查。 |
| FP32、FP64 | `%f/%F/%e/%E/%g/%G/%a/%A`；按double变参传递，`%lf`与`%f`等价。 |
| `HostLongDouble`，或可提升的FP32/FP64 | `%Lf/%LF/%Le/%LE/%Lg/%LG/%La/%LA`；普通浮点参数可提升为long double，但不会增加原始数值精度。扩展精度数据使用HostLongDouble，有效精度取决于CPU ABI。 |
| 字符代码或编译期单字符 | `%c`输出一个字节，按C规则取字符值，不是Unicode宽字符输出；多字节文本使用UTF-8字符串配合`%s`。 |
| 编译期字符串或运行时`HostString` | `%s`。 |
| `HostPointer`、`HostString`或Tensor元素的类型化地址 | `%p`，只打印地址。 |
| 格式控制 | 支持`-`、`+`、空格、`#`、`0`，固定宽度/精度及动态`*`；动态参数按有符号int32传递，int8/int16可以提升，int64及无符号值不能隐式替代。 |

Host不支持FP16/BF16独立标量、bool、任意Python对象或Tensor内容作为格式化参数，也不在CPU侧解引用NPU地址。负动态宽度表示左对齐，负动态精度按未指定精度处理；宽度是最小输出宽度，不自动截断较长数值。`h/hh`和`%c`可能截断数值，例如300使用`%hhd`得到44，321使用`%c`得到A，不会因超出字符或短整数范围而报错。字符串精度也按字节计数，例如`%.1s`可能只输出一个中文UTF-8字符的首字节，产生不完整文本；需要按编码边界控制截断长度。

Host运行时参数类型从`cannbotdsl`导出：

| 类型 | 数据和生命周期要求 |
| --- | --- |
| `HostString` | 接受str、bytes或ctypes字符指针/数组；str按UTF-8编码。自有或引用的存储须在调用期间有效，外部指针须非空且零结尾。运行时内嵌NUL按C字符串规则终止。 |
| `HostPointer` | 接受ctypes指针/数组、HostString或空指针，保留引用对象生命周期；不接受普通整数替代指针。 |
| `HostLongDouble` | 接受十进制文本、Decimal、数值或ctypes.c_longdouble，以16字节原生ABI传递。有效精度取决于CPU ABI；Python float不能恢复其原先已损失的精度。 |

编译时传入对应参数类型，执行时传入实例，复用Host参数ABI；这些类型不增加设备dtype或设备算术能力。

### 原有采集模式

标量支持bool、8/16/32/64位整数、FP16、BF16、FP32，不接受index、FP64或任意Python对象。采集式print对无符号标量可能按有符号值显示，例如uint8的255、uint32的4294967295显示为−1。需要无符号数值时使用格式化`%u/%llu`，或对对应Tensor使用dump接口。

Tensor支持UB/L1/L0C；该模式的解码支持与两个dump接口独立，不应根据dump的数据类型表推断旧采集也支持全部类型。

## 返回值说明

返回`None`。

Host和Kernel格式化输出不进入`get_debug_prints()`。只有旧采集模式使用`cannbotdsl.core.diag.debug.get_debug_prints()`读取结构化记录，并通过同模块的`clear_debug_prints()`清空已保存记录。

## 约束说明

### 显式使用DSL print与构图期警告

经过前端转换的`@host`、`@kernel`及继承其执行域的`@jit`中，若实际调用的是Python内置print，构图时发出`UserWarning`，定位到原始Python源文件和行号，并提示改用DSL接口：

> Python built-in print() in DSL code only executes during tracing; it will not print when the compiled Host/Kernel runs. Use 'from cannbotdsl import print' or 'cannbotdsl.print' for runtime output.

- 检查按被调用的函数对象识别，不是只检查是否写了import语句。未导入DSL print而调用内置print、通过别名调用内置print或显式调用`builtins.print`，都会触发提示；仅仅没有导入但未发生该调用，不触发提示。
- 默认行为是警告，不是强制报错。发出警告后仍执行原来的Python调用：常量立即显示，动态DSL值通常显示为对象或IR表示，`%d`等占位符不会按C printf规则自动替换。
- 该调用不生成Host或Kernel运行时打印指令，执行编译产物或AOT产物时不会重复输出这些内容。缓存命中且未重新构图时，也不会重新检查或发出此警告；已编译产物不会追溯产生构图期提示。
- 遵循Python标准warnings过滤策略。默认策略下，同一模块内同一源码位置提示一次；配置为ignore时不显示，配置为always时可重复显示，配置为error时会在原Python打印调用前抛出`UserWarning`异常并中断该调用。
- 普通Python代码中的print、正确使用DSL print，以及名称为print但并非内置print的自定义函数，不因本检查产生警告。普通Python辅助函数内部未经前端转换的调用不在该检查范围内。

### 程序入口要求

使用Host打印的程序，其运行时参数至少须包含一个符合约定的Tensor、TensorList或StructRef参数。仅有普通标量参数的入口在构造程序时会报`TypeError: staged provider calls require at least one canonical ... leaf`，尚未执行到print。只在Host中打印可以不发起Kernel，但不取消入口参数要求，详见[Host编译支持](../compilation/host/index.md)。

### 格式、地址和不支持范围

- Host/Kernel的格式必须为编译期字符串；不支持运行时格式指针、`%n`写回计数、按编号取参的位置格式或宽字符`%lc/%ls`。
- f-string、`.format()`或Python字符串插值不会自动变为DSL运行时格式化；涉及动态DSL值时可能在构图时失败，或把对象/IR表示固化为输出文本。例如`print(f"value={x}")`可能一直显示Integer对象信息。需要打印每次运行的值时，应使用格式字符串和独立参数。该情况不会触发内置print警告，因为调用目标仍是DSL print。
- 普通Kernel和SIMT的运行时字符串须在GM，SIMD VF须在UB；运行时字符串不会被隐式搬运。调用者保证指针地址、生命周期和零终止符有效。
- `%p`只显示指针，不能据此认为当前上下文可以读取该内存。`Tensor.ptr()`当前只支持GM/UB Tensor；对L1、L0C等直接调用该取址方法会在到达print之前报错。GM/UB Tensor的元素须先加载为标量，其他存储位置的数据须先通过合法路径取出。
- 纯AIC/AIV在对应核执行；MIX中的公共标量语句可能由两侧执行，依赖UB的打印只在AIV执行。SIMT每个线程独立输出，可用线程条件控制数量。

### 长度、缓冲和开销

- 不存在一个同时适用于Host、普通Kernel、SIMD VF和SIMT的统一“最多字符数”。Kernel的格式、参数及字符串内容都会占用CANN记录空间。
- SIMD VF与dump及原生断言共享2 KiB UB调测区。printf每条另占32字节管理信息，整条按8字节对齐，必须能完整放入记录区；不能将2 KiB直接当作可打印字符数。
- 新`readLen/finish`协议可在同一次VF中排空并复用空间，累计记录可以超过2 KiB；旧`pidx`协议保留批次限制。该能力不取消单条大小及GM日志容量限制。
- 含SIMD VF格式化print时总计预留10 KiB UB，其中2 KiB是格式及常量字符串临时区；用户手工存储也须避开保留区域。容量按UTF-8字节数计算，包含零终止符和自动追加的end。格式与常量字符串合计超出2 KiB时，翻译阶段拒绝，stderr提示`SIMD printf constant strings exceed the 2 KiB scratch budget`；Python侧可能只显示通用的`Translation to AscendC failed`。
- 常量区放得下不代表单条记录一定放得下。若运行时整条printf记录超过VF调测记录区容量，会置缓冲错误标志并跳过该记录；本次VF后续的print及dump也会被跳过，通常不抛Python异常。下一次重新初始化的VF调用可恢复输出。运行时`%s`内容同样计入记录大小，不能只检查格式字符串长度。
- CANN公开文档的SIMD默认打印缓冲限制为30 KB，SIMT默认缓冲为2 MB；相应配置字段为`simd_printf_fifo_size_per_core`和`simt_printf_fifo_size`。可配置范围、启动方式及SDK限制以实际环境为准，PyTorch调用及算子入图场景不能假定可调整SIMD缓冲。
- Host输出使用CPU标准输出，没有DSL VF的2 KiB限制，但仍受CPU运行库及存储有效性约束。

### 原有采集模式限制

- Tensor不接受GM本体；GM可打印先加载得到的受支持标量。Buffer/RingBuffer应先取得Tensor视图。
- Tensor单条最多采样1024字节，按物理存储顺序读取，不对非连续视图自动压紧，也不自动将NZ/ZN整体转换为ND。部分对齐空间可能占用采样预算。
- 不支持`end`或额外位置参数；`label/mode`仅用于采集模式。该模式使用隐藏GM捕获缓冲，触发Host回读和同步。
- 旧采集的Provider不能保存：调用`save()`会报`RuntimeError: debug-bearing ProviderCallable cannot be saved`。格式化print和dump本身没有这项限制，但与旧采集混用仍受该限制。

### 输出与构建

- `ASCENDC_DUMP`是编译期宏。使用`-DASCENDC_DUMP=0`编译并重新生成产物时，可关闭CANN设备格式化输出；仅设置同名Shell环境变量不能关闭打印，也不会改变已有产物。该宏不关闭普通Python、Host标准输出或独立的旧采集机制。
- 仅使用格式化打印支持编译缓存复用及AOT编译、保存、加载；与旧采集模式混用时仍受旧模式限制。
- Host调用发生于CPU程序运行时，Kernel文本由CANN异步消费。采集设备日志前应同步设备，宿主文本刷新仍可能稍晚，不保证跨核、跨线程或Host/Kernel之间的显示顺序。
- print会增加资源占用和执行耗时，不用于性能测量。

## 相关接口

- [dump_tensor](./dump_tensor.md)：打印连续物理存储区间。
- [dump_reg](./dump_reg.md)：打印SIMD VF矢量寄存器。
