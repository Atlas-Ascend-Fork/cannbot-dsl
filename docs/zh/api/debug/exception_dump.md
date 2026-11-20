# 异常 Tensor 落盘

## 功能说明

在 AIC/AIV 异常时，CANNBot-DSL 将当次任务的 Tensor 地址、shape、originShape、dtype 和 format 交给 CANN `acldumpSaveExceptionInfo`，由 CANN 生成 `.custom.<timestamp>` 二进制文件，供配套的 AI Core Error 工具解析。

本功能由运行时异常监测触发，无需在业务代码中主动调用保存接口。普通 Python 异常、参数类型检查和编译失败不会触发 AIC/AIV 异常回调。

使用前需要确认已安装的 CANNBot-DSL 版本包含异常 Tensor 导出适配。构建该运行时时需要提供兼容的官方 `acl/acl_dump.h`；运行时按库名加载 `libascend_dump.so`。SDK 头文件不兼容时，该适配会在构建时关闭。缺少兼容 SDK 或运行库时，正常导入、执行和原有诊断仍可用，异常报告记录该能力不可用的原因。

## 启用方法

先加载实际安装的 CANN 环境，在启动 Python、初始化 ACL/NPU **之前**设置异常 Dump 配置：

```bash
# 替换为实际 CANN 安装目录下的环境脚本。
source /path/to/cann/set_env.sh
export ASCEND_DUMP_SCENE=aic_err_brief_dump
export ASCEND_WORK_PATH="$PWD/.dump/cann"
mkdir -p "$ASCEND_WORK_PATH"
python your_operator.py
```

`your_operator.py` 替换为实际算子程序。正常执行不应生成异常 Tensor 文件，无需为了验证配置而在业务设备上主动制造故障。

| 配置 | 用途 | 设置时机 |
| --- | --- | --- |
| `ASCEND_DUMP_SCENE=aic_err_brief_dump` | 启用 CANN 的 AIC/AIV 异常 Dump 场景。 | 启动 Python、初始化 ACL/NPU 之前。 |
| `ASCEND_WORK_PATH` | 配置 CANN 工作目录；最终落盘目录以 CANN 接口返回值为准。 | 启动 Python、初始化 ACL/NPU 之前。 |
| `CANNBOTDSL_AIC_ERROR_DEBUG=1` | 用于编译行号，不是 CANN 异常 Dump 开关。 | 编译算子之前。 |
| `CANNBOTDSL_AIC_ERROR_DEBUG_DIR` | 配置 DSL 异常报告目录，与 CANN 目录独立。 | 启动算子程序之前。 |

实际 CANN 目录通过 `acldumpGetExceptionInfoPath` 查询。配置路径可能受 CANN 环境变量优先级影响，应以接口返回路径为准；路径查询失败不会阻止保存调用。

`ASCEND_WORK_PATH` 与 [AscendC 源码归档](../../guide/ascendc-dump.md)使用的 `CANNBOTDSL_DUMP_DIR` 用途不同。后者保存编译生成的源码，不保存异常时的 Tensor 数据。

## Tensor 范围

| 项目 | 支持范围与处理方式 |
| --- | --- |
| 布局和维数 | 连续 ND Tensor，逻辑与物理 shape 一致，rank 为 0～25。 |
| 数据类型 | bool，8/16/32/64 位有符号和无符号整数，float16、bfloat16、float32、float64。 |
| 参数角色 | `input`、`inout`、`unknown` 使用标准 input 索引；纯 `output` 使用标准 output 索引。两类索引分别计数。 <!-- codespell:ignore inout --> |
| 不可导出参数 | unused、非连续、负 stride、空 Tensor、私有格式、未知类型、超维，以及 ABI 或地址校验失败的参数被跳过，并记录原因；其他有效 Tensor 仍可保存。 |
| 片上数据 | 不导出内部 UB/L1 或未登记的 workspace。 |

CANN 保存接口不接收 strides，不能将任意非连续 Tensor 当作连续 Tensor 导出。发生故障后，运行时也不会提交额外 Kernel 整理数据。

读写型 Tensor 保存异常时的一份观测值。数据**不保证是执行前输入，也不保证包含完整输出或可以用于复现故障**。内存分配范围校验不能证明 Tensor 仍存活、未被复用或未被其他 stream 修改。

## 异常处理与任务关联

运行时先精确关联异常任务，再从该次提交的参数字节中读取 Tensor 地址和动态 shape，在原进程中切换到提交时的 context、核对 device 后调用 CANN 保存接口，完成后恢复调用前的 context。

保存发生在 DSL 诊断文件写入之前，不依赖 recovery 目录、快照、旧输入复制或 `aclrtMemcpy` 成功。离线报告进程只读取文件，不执行设备侧保存。尚未完成精确关联的 pending 事件会等待原有任务关联；ambiguous 事件不会猜测参数。同线程递归回调受保护。

## 输出文件与状态

```text
<CANN 返回目录>/
  dsl_exception_<pid>_<session>_<sequence>.custom.<timestamp>
  exception-dump-<pid>-<session>-<sequence>.json

<DSL 报告目录>/process-<pid>-*/
  snapshot-<sequence>.json
  error-*.json / error-*.log
```

| 产物 | 用途 |
| --- | --- |
| `.custom.<timestamp>` | CANN 标准异常 Tensor 二进制，交给配套工具解析。 |
| `exception-dump-*.json` | 辅助结果，记录 API 原始返回码、路径查询码、事件身份、Tensor 描述、标准索引和跳过原因。 |
| `snapshot-*.json`、`error-*.json/.log` | DSL 异常快照和定位报告，提供事件关联及辅助结果路径。 |
| 原有输入 `.bin/.pt` | DSL 输入采集的其他诊断产物，不能代替 CANN 标准文件。 |

每次事件使用独立请求名称。`user_tag` 包含 artifact、kernel 以及 `in0:arg0,out0:arg2` 等索引映射；标准 input/output 索引不等同于 DSL 参数序号。`argsOffset=0` 表示独立自定义数据集，不表示获得了原 Kernel 的设备参数 ABI 偏移。这些数据不会自动成为原算子的复跑输入。

| 状态 | 含义 |
| --- | --- |
| `deferred` | 尚未精确关联任务，未调用保存。 |
| `skipped` | 任务或参数不满足导出条件。 |
| `unavailable` | SDK、库或 context 等能力不可用。 |
| `failed` | CANN 保存返回非零，保留原始返回码。 |
| `api_success` | 保存返回零，尚未确认最终文件。 |
| `file_confirmed` | CPU 报告侧确认唯一、非空的对应文件存在。 |
| `file_unconfirmed` | API 返回零，但缺少可靠路径或未确认唯一文件。 |
| `invalid_result` | 辅助结果损坏或与事件身份不匹配。 |

文件存在不代表内容已经解析验证。若 CANN 路径查询失败，或该目录无法写入辅助 JSON，运行时不会改到 DSL 目录写该辅助文件；异常快照仍可保留保存状态和返回码，stderr 仍输出 API 状态、请求名称、路径和标签。进程退出或辅助记录未生成时，可能只有 CANN 文件或 stderr，不能仅凭缺少辅助结果判定保存成功或失败。

## 解析与验证

使用目标 CANN 安装包附带的 `msaicerr` 解析标准文件：

```bash
python "$ASCEND_HOME_PATH/tools/msaicerr/msaicerr.py" \
  -d /absolute/path/dsl_exception_....custom.123 \
  -out /absolute/path/decoded
```

部分安装包将工具放在 CANN 根目录，请将入口替换为实际安装路径。解析后核对 Tensor 的 shape、dtype、user tag 和数据值。单文件解析成功只验证该文件的解析结果，不能据此判定完整故障材料自动采集、定位和算子复跑均已通过。

原框架仓库中的回归测试通过真实异常回调调用已安装 CANN 的保存接口，并核对事件身份、标准文件、参数映射和解码数据。这些测试及其 `build/dev-env.sh`、`test/cannbotdsl/...` 路径不属于本算子仓库，不能作为本仓可直接运行的命令。

需要在框架仓库验证时，应分别检查正常执行不产生异常 Dump，以及故障后的文件内容和事件关联。真实故障测试仅在空闲、独占的测试设备上显式启用，不作为普通业务运行或常规 CI 的步骤。具体测试入口和启用方式以对应框架版本为准。

## 约束说明

- CANN 接口、驱动或设备故障可能导致数据不可读，不保证所有故障都能生成完整文件。
- CANN 保存接口没有已确认的取消或超时能力，不能在回调中强行终止该调用。
- Graph replay 和更新参数场景仍受原有任务快照机制限制，不能据此推定获取了新的完整参数快照。
- 标准文件、辅助 JSON 和 DSL 报告应按事件身份关联，不能混用不同进程或不同故障事件的产物。

## 相关资料

- [调试与性能分析](./index.md)
- [调试与精度验证](../../guide/debugging.md)
- [dump_tensor](./dump_tensor.md)：设备代码主动输出 Tensor 数据。
- [dump_reg](./dump_reg.md)：设备代码主动输出矢量寄存器。
- [acldumpSaveExceptionInfo 官方文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/API/runtimeapi/aclcppdevg_03_1289.html)。实际 ABI 以目标环境的官方头文件为准。
