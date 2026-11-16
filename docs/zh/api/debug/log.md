# log 通用日志接口

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

在算子运行时按级别记录日志，帮助查看参数、分支和执行阶段。`error`记录ERROR日志并上报SDK最近错误；`error_without_report`仅记录ERROR日志。

| 接口 | 级别 | 使用场景 | SDK 行为 |
| --- | --- | --- | --- |
| `debug` | DEBUG / 0 | 细粒度标量值、分支和循环诊断 | `OP_LOGD` |
| `info` | INFO / 1 | 输入摘要、正常阶段和状态 | `OP_LOGI` |
| `warn` | WARNING / 2 | 可继续执行的异常或降级提示 | `OP_LOGW` |
| `error` | ERROR / 3 | 需要写入 SDK 最近错误的诊断 | `OP_LOGE`，内部传入错误码 561000 |
| `error_without_report` | ERROR / 3 | 补充错误日志，保留已有最近错误 | `OP_LOGE_WITHOUT_REPORT` |
| `event` | EVENT / 16 | 重要事件或阶段标记 | `OP_EVENT`；接口本身不增加设备同步 |

## 函数原型

```python
cannbotdsl.log.debug(fmt, *args, tag="") -> None
cannbotdsl.log.info(fmt, *args, tag="") -> None
cannbotdsl.log.warn(fmt, *args, tag="") -> None
cannbotdsl.log.error(fmt, *args, tag="") -> None
cannbotdsl.log.error_without_report(fmt, *args, tag="") -> None
cannbotdsl.log.event(fmt, *args, tag="") -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `fmt` | 输入 | 编译期Python字符串，不能含NUL。支持`%d %i %u %o %x %X %f %F %e %E %g %G %s %%`。宽度、精度最大1024；不支持`%p`、`%ld`或动态`*`宽度。字面百分号写成`%%`。 |
| `args` | 输入 | 按格式顺序传入编译期常量或运行时DSL数值标量。动态整数最多64位；Host动态浮点支持float32/float64，设备支持float32，窄浮点提升到float32。`%s`只接受编译期字符串。不接受Tensor、向量寄存器、运行时字符串；类型和数量须与格式匹配。 |
| `tag` | 输入 | 仅限关键字传参，编译期字符串，默认空串，不能含NUL。添加诊断标签，不改变SDK子模块。 |

框架自动记录文件、行号、调用角色及当前DSL函数名。ERROR上报使用固定错误码561000。EVENT的日志级别为16，日志级别与模块标识是独立字段。`tag`可使用`COMPILE = "compile"`。

## 返回值说明

返回`None`。

## 约束说明

- 仅支持在`@host`、`@kernel`及相应执行域的`@jit`中调用。在普通Python代码中直接调用会报错。
- 错误日志仅记录错误，不检查输入、不抛出异常、不停止Kernel，也不返回失败状态。算子需通过分支处理非法输入，避免继续执行导致越界。
- Kernel外层标量控制流支持设备日志。SIMD VF内部、SIMT线程级日志及SIMT算子外层组合的支持情况尚未明确，请勿用于每个lane或thread的并发打印。
- 设备日志要求在编译期确定启动block数。每个日志调用点在每个core的缓冲槽中保留最近64条记录，每条最多包含128个动态标量；旧记录被覆盖时会输出提示。Host同步回读后，按调用点和core输出日志，不保证跨核顺序。时间戳为Host输出日志的时间。启用设备日志会增加回读和同步开销。

格式规则与[print](./print.md)不同，两者的参数不能直接互换。设备日志沿用debug-buffer的生命周期；AOT保存和加载受框架现有能力限制。

## 调用示例

以下示例拷贝一个元素并记录六类日志。当`count`为负数时，仅记录参数错误，不启动Kernel。将代码保存为`log_example.py`后运行。

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
    if count < 0:
        cbd.log.error_for_invalid_value("LogDemo", "count", count, "nonnegative")
        # 通过分支跳过非法输入对应的计算。
    else:
        cbd.log.debug("count=%d", count)
        cbd.log.info("host begins")
        cbd.log.warn("demonstration warning")
        cbd.log.error("demonstration error")
        cbd.log.error_without_report("additional diagnostic")
        copy_one[1](x, y)
        cbd.log.event("host completed")


def main():
    torch.npu.set_device(0)
    x = torch.tensor([[3.0]], device="npu", dtype=torch.float32)
    y = torch.zeros_like(x)
    spec = cbd.TensorSpec((1, 1), cbd.dtypes.float32)
    program = cbd.compile(entry, spec, spec)
    try:
        program(x, y, 1)
        torch.npu.synchronize()
        torch.testing.assert_close(y.cpu(), x.cpu())
        y.zero_()
        program(x, y, -1)
        torch.npu.synchronize()
        torch.testing.assert_close(y.cpu(), torch.zeros((1, 1)))
    finally:
        program.close()
    print("log example completed")


if __name__ == "__main__":
    main()
```

示例运行结束后输出以下消息。日志前缀、文件行号及核编号随运行环境变化。

```text
log example completed
```

### 日志输出配置

启动Python前，先加载CANN的`set_env.sh`。如需输出DEBUG、EVENT日志并显示到终端，设置以下环境变量：

```bash
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_GLOBAL_EVENT_ENABLE=1
export ASCEND_SLOG_PRINT_TO_STDOUT=1
python log_example.py
```

如需将日志写入plog文件，将`ASCEND_SLOG_PRINT_TO_STDOUT`设为`0`。可在CANN配置的日志根目录中，通过`CANNBOT DSL`或示例消息查找日志。具体路径和文件名取决于运行用户及CANN配置。

原生后端通过`libcannbotdsl_helper.so`调用ACLNN的`OP_LOG*`接口。日志使用SDK的OP/NNOP模块，并包含`[CANNBOT DSL]`标识。`CANNBOT_DSL = 16`是框架内部标识，并非SDK中的独立模块。运行库默认自动查找helper库，也可在首次调用前，通过`CANNBOTDSL_ACLNN_LOG_LIBRARY`指定其绝对路径。

helper库或所需符号不可用时，日志改用C++ printf输出。此时不保证日志写入plog或上报SDK最近错误；终端显示日志并不表示日志已写入CANN日志系统。
