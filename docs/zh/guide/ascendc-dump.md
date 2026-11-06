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
| `CANNBOTDSL_DUMP_ASCENDC` | 默认关闭；设为 `1` 时保存源码。 |
| `CANNBOTDSL_DUMP_DIR` | 保存目录，默认是当前工作目录下的 `.dump`；设置目录本身不会开启归档。 |

其他编译调试选项见[编译选项与产物观察](./compiler-options.md)。

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

## 使用注意

- 只保存本次编译实际生成的源码。加载已有二进制、运行已创建的可调用对象或Graph replay时，若没有重新编译，就不会产生新归档；需要补充源码时，应在构建阶段开启后重新编译。
- `CANNBOTDSL_REUSE_ASCENDC=1` 与源码归档互斥。Native `require` 部署模式不会为了归档而回退到在线编译，需在构建环境保存源码。
- `kernel.asc` 是生成的AscendC代码；元数据中的Python文件路径用于确认来源，不提供两者的逐行映射。
- 开启后会增加源码生成和文件写入的编译开销，并影响缓存复用路径；不会插入kernel调试指令或同步操作。关闭时跳过归档。
- 目录不可写或磁盘空间不足时会抛出编译异常。文件逐个原子替换、元数据最后写入；中途失败后应以有效的 `metadata.json` 和源码摘要确认归档是否完整。
- 归档目录需自行保存和清理，不会自动打进算子包；交付源码时请同时保留该目录。

## 与运行时调试输出的区别

本功能在编译阶段归档源码，不读取输入Tensor的数据，也不记录设备运行时周期。
[print_time_stamp](../api/debug/print_time_stamp.md)在设备执行时记录周期和PC，其原生二进制保存路径由CANN Dump配置及相关环境变量控制，不使用`CANNBOTDSL_DUMP_DIR`。
