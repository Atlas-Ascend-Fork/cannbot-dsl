# AscendC源码归档

编译算子包或进行即时编译（Just-In-Time，JIT）时，可将生成的AscendC源码按算子和shape/dtype变体分别保存，避免不同算子的入口都叫 `run` 时互相覆盖。

## 开启方法

在编译算子包或启动在线编译进程前设置：

```bash
export CANNBOTDSL_DUMP_ASCENDC=1
export CANNBOTDSL_DUMP_DIR="$PWD/operator-dump"
python your_compile_or_jit_script.py
```

最后一行替换为实际构建或在线编译命令，无需修改kernel代码。

| 配置 | 说明 |
| --- | --- |
| `CANNBOTDSL_DUMP_ASCENDC` | 统一dump开关，默认关闭；设为`1`时归档源码，并在运行时保存输入Tensor快照和tiling信息。 |
| `CANNBOTDSL_DUMP_DIR` | 保存目录，默认是当前工作目录下的 `.dump`；设置目录本身不会开启归档。 |

其他编译调试选项见[编译选项与产物观察](./compiler-options.md)。

同一开关还会把[输入Tensor快照](input-tensor-dump.md)保存到`inputs/`，把[Tiling结果](tiling-dump.md)保存到`tiling/`，均使用同一`CANNBOTDSL_DUMP_DIR`。

## 文件目录

```text
operator-dump/
  run.asc
  asc/<Python模块.限定函数名>-<身份哈希>/
    x-32x64-fp16__y-32x64-fp16--<变体哈希>/
      kernel.asc
      metadata.json
      host.cpp
```

- 算子目录使用Python模块和限定函数名，区分不同类或模块中的同名入口。
- 变体目录包含参数名、shape、dtype和哈希；同一动态shape契约使用同一目录，维度显示为符号或表达式。较长名称会截断可读部分，保留哈希。
- `kernel.asc` 是本次入口编译生成的AscendC源码，可能包含多个kernel和辅助函数。
- `metadata.json` 保存算子入口、完整参数契约、编译目标和源码摘要；`host.cpp` 仅在源码导出流程生成独立host源码时保存。
- 根目录的 `run.asc` 保留最近一次源码，仍可能被同名入口覆盖；查看各个变体请使用 `asc/` 子目录。

先按算子名、shape和dtype找到目录，再核对 `metadata.json` 中的 `contract`（完整参数契约）、`target`（编译目标）和 `operator`（算子入口），最后打开同目录的 `kernel.asc`。

## 普通dump与异常诊断目录

普通源码与执行前Tensor快照属于手动开启的调试输出，默认不写到AIC异常目录。默认目录结构如下；只创建本次实际需要的分支：

```text
.dump/
  asc/                        # 源码：按算子入口和编译变体组织
  inputs/                     # 输入快照：按入口、进程和调用组织
  tiling/                     # Tiling信息：由统一开关控制
  <入口函数名>.asc            # 最近一次编译源码，兼容旧复用流程
  aic_error/                  # 只有收到异常后才创建的诊断现场
```

| 内容 | 默认位置 | 创建时机 | 路径配置 |
| --- | --- | --- | --- |
| 普通AscendC源码 | `.dump/asc/` | 开关开启且本次实际生成源码时 | `CANNBOTDSL_DUMP_DIR` |
| 执行前Tensor快照 | `.dump/inputs/` | 开关开启后调用算子时；Graph capture只写跳过说明 | `CANNBOTDSL_DUMP_DIR` |
| Tiling信息 | `.dump/tiling/` | 开关开启后提交kernel前 | `CANNBOTDSL_DUMP_DIR` |
| 异常kernel、参数、寄存器、源码和异常时输入 | `.dump/aic_error/process-<PID>-*/` | 收到异常后 | `CANNBOTDSL_AIC_ERROR_DEBUG_DIR` |

AIC目录行为适用于包含“诊断仅异常时落盘”更新的框架包，详见[异常诊断的使用前准备](aic-error-debug.md#使用前准备)。未开启普通dump和其他调试dump、且没有异常时，不会因这些功能创建`.dump`；已经存在的旧目录不会自动清理。`CANNBOTDSL_AIC_ERROR_DEBUG`控制编译行号及Python来源映射，不会打开普通ASC/PT dump。

可将两类输出分开配置：

```bash
export CANNBOTDSL_DUMP_DIR="$PWD/operator-dump"
export CANNBOTDSL_AIC_ERROR_DEBUG_DIR="$PWD/aic-reports"
```

设置目录变量本身不会打开普通dump；如需普通输出，再设置`CANNBOTDSL_DUMP_ASCENDC=1`。框架分别使用上述两个根目录；若自行把两者设置为同一路径，输出也会使用该路径。输入目录的每层名称见[输入Tensor快照](input-tensor-dump.md#保存范围与文件组织)，异常材料见[异常诊断输出文件](aic-error-debug.md#输出文件)。

## 使用注意

- 只保存本次编译实际生成的源码。加载已有二进制、运行已创建的可调用对象或Graph replay时，若没有重新编译，就不会产生新归档；需要补充源码时，应在构建阶段开启后重新编译。
- `CANNBOTDSL_REUSE_ASCENDC=1` 与源码归档互斥。Native `require` 部署模式不会为了归档而回退到在线编译，需在构建环境保存源码。
- `kernel.asc` 是生成的AscendC代码；元数据中的Python文件路径用于确认来源，不提供两者的逐行映射。
- 源码归档会增加编译和文件写入开销，并影响缓存复用路径。统一开关同时启用的输入快照会进行设备到CPU的阻塞复制，tiling会在提交前落盘；仅建议定位问题时开启。不会插入kernel调试指令，关闭时跳过归档。
- 目录不可写或磁盘空间不足时会抛出编译异常。文件逐个原子替换、元数据最后写入；中途失败后应以有效的 `metadata.json` 和源码摘要确认归档是否完整。
- 归档目录需自行保存和清理，不会自动打进算子包；交付源码时请同时保留该目录。

## 与运行时调试输出的区别

源码归档步骤本身只保存编译源码，不读取输入Tensor数据或记录设备运行时周期；同一开关启用的输入快照会另行复制数据。
[print_time_stamp](../api/diag/print_time_stamp.md)在设备执行时记录周期和PC，其原生二进制保存路径由CANN Dump配置及相关环境变量控制，不使用`CANNBOTDSL_DUMP_DIR`。
