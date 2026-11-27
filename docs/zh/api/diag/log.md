# log通用日志接口

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

按DEBUG、INFO、WARNING、ERROR和EVENT级别记录诊断信息，用于查看算子参数、分支和执行阶段。同一接口支持普通Python、`@host`、`@kernel`及相应执行域的`@jit`；`error`在DSL的正常CANN路径中还上报SDK最近错误，`error_without_report`只记录ERROR。

| 接口 | 日志级别 | 用途 | DSL正常CANN路径 |
| --- | --- | --- | --- |
| `debug` | DEBUG / 0 | 标量值、分支和循环诊断。 | `OP_LOGD`。 |
| `info` | INFO / 1 | 输入摘要、正常阶段和状态。 | `OP_LOGI`。 |
| `warn` | WARNING / 2 | 可继续执行的异常或降级提示。 | `OP_LOGW`。 |
| `error` | ERROR / 3 | 错误诊断，同时上报最近错误。 | `OP_LOGE`，框架传入错误码561000。 |
| `error_without_report` | ERROR / 3 | 补充错误日志，保留已有最近错误。 | `OP_LOGE_WITHOUT_REPORT`。 |
| `event` | EVENT / 16 | 重要事件或阶段标记；调用本身不主动执行设备同步，设备日志回读仍涉及同步和数据搬运开销。 | `OP_EVENT`。 |

## 导入方式

通用日志和错误模板接口统一支持以下两种导入方式。

方式一：导入包，通过`cbd.log.xxx`调用接口。

```python
import cannbotdsl as cbd

cbd.log.info("size=%d", 16, tag="check")
```

方式二：从`utils`导入日志模块，通过`log.xxx`调用接口。

```python
from cannbotdsl.utils import log

log.info("size=%d", 16, tag="check")
```

两种方式引用同一日志模块，接口参数和行为一致。迁移时统一使用上述入口，不通过旧物理模块路径`import cannbotdsl.log`或内部实现模块导入接口。

## 函数原型

```python
import cannbotdsl as cbd

cbd.log.debug(fmt: str, *args: str | int | float | cbd.ScalarValue, tag: str = "") -> None
cbd.log.info(fmt: str, *args: str | int | float | cbd.ScalarValue, tag: str = "") -> None
cbd.log.warn(fmt: str, *args: str | int | float | cbd.ScalarValue, tag: str = "") -> None
cbd.log.error(fmt: str, *args: str | int | float | cbd.ScalarValue, tag: str = "") -> None
cbd.log.error_without_report(fmt: str, *args: str | int | float | cbd.ScalarValue, tag: str = "") -> None
cbd.log.event(fmt: str, *args: str | int | float | cbd.ScalarValue, tag: str = "") -> None
```

以上为接口签名说明；`tag`只能以关键字传入。[ScalarValue](../data-structures/data-types.md#标量值)表示DSL数值标量，具体支持类型见参数说明。

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `fmt` | 输入 | 格式字符串，类型为`str`，不能含NUL。普通Python在调用时读取；DSL中必须为编译期常量。支持`%d %i %u %o %x %X %f %F %e %E %g %G %s %%`，以及`- + 空格 # 0`标志和固定宽度、精度；宽度、精度最大为1024。不支持`%p`、`%ld`、动态`*`宽度或精度。即使无参数，字面百分号也须写成`%%`。 |
| `args` | 输入 | 按转换项顺序传入，数量和类型须与`fmt`匹配。普通Python的整数、浮点、字符串分别使用`int`、`float`、`str`。DSL支持编译期常量和运行时数值标量；`%s`只接受编译期字符串。动态整数最多64位；Host动态浮点支持float32/float64，设备支持float32，窄浮点提升到float32。不接受Tensor、矢量寄存器或运行时字符串。 |
| `tag` | 输入 | 可选，诊断标签，类型为`str`，默认值为`""`，不能含NUL；DSL中必须为编译期常量。标签写入消息前缀，不改变SDK日志模块或子模块。 |

有符号整数格式的Python常量范围为`[-2**63, 2**63-1]`；无符号格式允许`[-2**63, 2**64-1]`，负数按64位无符号数转换。数值转换的输出最多保留1024个字符。浮点格式不将Python `int`隐式作为`float`处理。

调用方无需传入模块号或错误码。ERROR使用框架绑定的561000错误码；EVENT的原生级别为16，与模块标识无关。框架自动记录用户文件、行号、当前函数名、执行角色及设备日志的block/subblock。

## 返回值说明

返回`None`。接口不返回算子执行状态。

## 约束说明

- 调用方须检查业务条件，并自行分支、返回或跳过计算。ERROR和[错误模板](./error_message.md)负责记录诊断，不检查业务条件、不抛出业务异常或停止Kernel。
- 已验证范围为普通Python、Host及Kernel外层标量控制流。未验证范围包括SIMD VF内部日志、SIMT线程级日志及SIMT算子外层与日志的完整组合；SIMT不具备每个thread/lane的独立日志槽。打印矢量寄存器应使用[dump_reg](./dump_reg.md)。
- 设备日志要求编译期可确定启动block数。每个调用点在每个block/subblock的独立槽中保留最近64条记录，每条最多128个动态标量；超出保留容量时覆盖旧记录并给出提示，槽位布局根据目标芯片能力区分AIC/AIV。
- 设备日志按调用点、block/subblock回放，不保证跨核的先后顺序，时间戳为Host输出时间；日志回读增加同步和数据搬运开销。设备日志沿用debug-buffer生命周期，其AOT保存/加载能力受框架当前实现限制。

### 日志输出配置

DSL运行前先加载所用CANN安装目录的`set_env.sh`。以下配置同时开启三个路径的DEBUG、EVENT和终端输出：

```bash
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_GLOBAL_EVENT_ENABLE=1
export ASCEND_SLOG_PRINT_TO_STDOUT=1
```

| 环境变量 | 普通Python | DSL + CANN | DSL + printf回退 |
| --- | --- | --- | --- |
| `ASCEND_GLOBAL_LOG_LEVEL` | `0/1/2/3/4`分别为DEBUG/INFO/WARNING/ERROR/关闭普通日志；有效值优先于Logger级别。未设置、空值或非法值时使用Logger级别；未显式配置Logger时默认ERROR，保留用户显式设置的级别。 | 由CANN日志系统控制。当前验证的SDK在未设置、空值或非法值时默认ERROR。 | 使用同样的五档阈值；未设置、空值或非法值时默认ERROR。 |
| `ASCEND_GLOBAL_EVENT_ENABLE` | `0`关闭EVENT，`1`开启。无有效开关时不主动关闭；存在有效普通日志配置时EVENT独立于该阈值，否则遵循Python Logger级别。 | 由CANN独立控制EVENT。 | `0`关闭，其他值或未设置时开启；独立于普通日志阈值。 |
| `ASCEND_SLOG_PRINT_TO_STDOUT` | `1`令默认Handler输出到stdout，`0`关闭默认Handler打屏，其他值或未设置时输出到stderr；不改变用户Handler。 | `1`打屏，`0`使用CANN文件日志路径，按SDK配置写入plog。 | `0`关闭输出，其他值或未设置时输出到stdout；不创建plog。 |

普通Python专用Logger为`cannbotdsl.operator`，默认ERROR。因此在未显式配置Logger、全局等级未设置且EVENT开关未设置时，DEBUG、INFO、WARNING和EVENT均被过滤，只输出ERROR；显式设置`ASCEND_GLOBAL_EVENT_ENABLE=1`可独立开启EVENT。

普通Python每次调用读取上述变量；不接管`ASCEND_MODULE_LOG_LEVEL`、`ASCEND_PROCESS_LOG_PATH`等SDK配置。Python Logger默认不向父Logger传播，不修改root Logger。可使用标准`logging`配置级别、过滤器、Handler或Formatter；`logging.disable`、Logger.disabled、用户Handler的级别与过滤器仍生效。需要Python文件日志时自行配置`logging.FileHandler`，这不属于CANN plog。

DSL正常路径自动查找`libcannbotdsl_helper.so`，也可在首次调用前用`CANNBOTDSL_ACLNN_LOG_LIBRARY`指定其绝对路径。缺库或缺符号时回退C++ printf。CANN plog路径由SDK配置决定，可在日志根目录按`CANNBOT DSL`或消息内容检索。

### 日志格式

三种路径的默认输出使用同一字段顺序和格式结构；时间、进程/线程ID、用户位置及后端位置取运行时真实值。自定义Python Formatter后，外层格式由用户配置决定。

```text
[LEVEL] OP(pid,process):YYYY-MM-DD-HH:mm:ss.mmm.uuu [backend_file:line][NNOP][backend_function][tid] errno[561000] OpName:[function] [source_file:line][CANNBOT DSL][tag][@role] message
```

`errno[561000]`仅出现在ERROR。普通Python的角色为`python`，DSL为`host`、`kernel`或`jit`；设备域在角色之后增加`[block=<编号>,subblock=<编号>]`。同一有效参数值采用相同消息格式，但不同浮点精度的舍入和元信息可能不同。三种路径的普通日志默认阈值均为ERROR；EVENT的默认过滤行为见上表。此接口的格式规则与[print](./print.md)不同，不能直接互换参数。

## 调用示例

以下完整示例在普通Python、Host和Kernel中记录日志，演示六类通用接口，拷贝一个元素并在Host侧校验。将代码保存为`log_example.py`，加载CANN环境并设置上述日志变量后执行`python log_example.py`。

```python
import torch
import torch_npu  # noqa: F401
import cannbotdsl as cbd


@cbd.kernel
def copy_one(x: cbd.Tensor, y: cbd.Tensor):
    cbd.log.info("copy begins", tag="copy")
    y[0, 0] = x[0, 0]


@cbd.host
def entry(x: cbd.Tensor, y: cbd.Tensor, count: int = 1):
    cbd.log.debug("count=%d", count, tag="copy")
    cbd.log.info("host begins", tag="copy")
    cbd.log.warn("demonstration warning", tag="copy")
    cbd.log.error("demonstration error", tag="copy")
    cbd.log.error_without_report("additional diagnostic", tag="copy")
    copy_one[1](x, y)
    cbd.log.event("host completed", tag="copy")


def main():
    cbd.log.info("python begins", tag="copy")
    torch.npu.set_device(0)
    x = torch.tensor([[3.0]], device="npu", dtype=torch.float32)
    y = torch.zeros_like(x)
    spec = cbd.TensorSpec((1, 1), cbd.dtypes.float32)
    program = cbd.compile(entry, spec, spec)
    try:
        program(x, y, 1)
        torch.npu.synchronize()
        torch.testing.assert_close(y.cpu(), x.cpu())
    finally:
        program.close()
    print("log example passed")


if __name__ == "__main__":
    main()
```

预期结果：除环境配置允许的日志外，程序稳定输出：

```text
log example passed
```

例如INFO日志包含`OpName:[entry]`、`[CANNBOT DSL][copy][@host] host begins`；设备日志包含`[block=<编号>,subblock=<编号>] copy begins`。ERROR日志含`errno[561000]`。时间、源码行号和物理执行编号使用实际运行值，不以固定值作为校验条件。
