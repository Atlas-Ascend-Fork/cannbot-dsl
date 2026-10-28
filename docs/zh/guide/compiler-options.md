# 编译选项与产物观察

这一章回答两个问题：**哪些开关会影响生成的代码**，以及**怎么看到编译结果**。前者关系到正确性与可复现性，后者关系到你能不能亲自验证[总览](./overview.md)里那句承诺——「代码写成什么样，设备上大致就执行成什么样」。

::: info 本章的依据
下面列出的环境变量集中定义在 `cannbotdsl/core/utils/env.py`，官方 API 文档站没有对应页面。它们可以放心用于调试，但不要写进生产脚本的硬依赖——版本间可能变化。
:::

## 编译链路回顾

```text
你的 Python 源码（@host / @jit / @kernel）
   │
   ├─① 前端识别：读语法结构，判定每个 if / for 是编译期还是设备期
   │        ↑ CANNBOTDSL_DUMP_FRONTEND / FRONTEND_LINE_INFO
   │
   ├─② 构图：Python 解释器执行函数体，记录搬运与计算为 CANNIR（MLIR）
   │
   ├─③ Pass pipeline：IR 降级与优化
   │        ↑ CANNBOTDSL_DUMP_MLIR / PRINT_MLIR / CANNIR_PIPELINE
   │
   ├─④ 翻译：CANNIR → AscendC 源码
   │        ↑ CANNBOTDSL_DUMP_ASCENDC / PRINT_ASCENDC / REUSE_ASCENDC
   │
   ├─⑤ 毕昇编译：AscendC → dav-3510 二进制
   │        ↑ --cce-* 选项（见下），CANNBOTDSL_BARALL
   │
   └─⑥ 加载并在 NPU 上执行
            ↑ cbd.print / dump_tensor / dump_reg / msprof
```

每一步都有观察手段，下面逐段讲。

## 看编译产物

### 转储 MLIR 与 AscendC 源码

这是最直接的「看生成了什么」。两个开关：

```bash
export CANNBOTDSL_DUMP_MLIR=1        # 每个 Pass 之后写 <function>.mlir
export CANNBOTDSL_DUMP_ASCENDC=1     # 翻译完成后写 <function>.asc
export CANNBOTDSL_DUMP_DIR=./mydump  # 可选，默认是当前目录下的 .dump/
python my_op.py
```

`.asc` 文件就是最终交给毕昇编译器的 AscendC 源码——循环展成了什么样、同步插在哪里、tile 常量固化成了多少，全部可读。这是验证「显式 DSL 的承诺是否兑现」最硬的证据。

想直接打到终端而不落盘：

| 变量 | 作用 |
| --- | --- |
| `CANNBOTDSL_PRINT_MLIR=1` | 在各 Pass 之间打印 MLIR |
| `CANNBOTDSL_PRINT_ASCENDC=1` | 以 DEBUG 日志输出最终的 AscendC 源码（需同时放开日志级别） |

::: warning 这些开关会旁路两级编译缓存
`DUMP_MLIR` / `DUMP_ASCENDC` / `PRINT_*` / `PIPE_STAGE` / `REUSE_ASCENDC` 打开时会跳过内存与磁盘缓存，所以每次都真的重新编译一遍——不会出现「开了 dump 却因为命中缓存而没有文件」的情况，代价是慢。测性能前记得关掉。
:::

### 在指定阶段停下来

```bash
export CANNBOTDSL_PIPE_STAGE=transform   # 或 translate / compile
```

只跑到某个阶段就停，用于定位「是 Pass 阶段出的问题还是翻译阶段出的问题」。

### 改写中间产物再编译

```bash
export CANNBOTDSL_REUSE_ASCENDC=1
```

用已有的 `<dump-dir>/<function>.asc` 替换翻译结果喂给毕昇。手工改几行 AscendC 验证猜想时很有用。

三条约束：

- 它与 `CANNBOTDSL_DUMP_ASCENDC` **互斥**，避免覆盖你手工改过的源码。
- 文件缺失会抛 `FileNotFoundError`，不会静默回退。
- 复用时会校验 dump 文件里的 `// CANNBOTDSL_HOST_ABI_VERSION=N` 标记。那是生成源码里的一行注释，**不是可设置的环境变量**；不匹配时需要重新 dump。
- 复用**不跳过** tracing、Pass 和翻译，只是把翻译的输出换掉。

### 覆盖 Pass pipeline

```bash
export CANNBOTDSL_CANNIR_PIPELINE='builtin.module(canonicalize,cse,func.func(...))'
```

锚定到 `FuncOp` 的 Pass 必须写成 `func.func(...)`，否则 PassManager 解析失败，只记一条 warning 就回退到 `builtin.module(canonicalize,cse)`。**产物看起来没变换时先查这条日志。** 这个变量进编译缓存键，所以不同 pipeline 不会互相命中。

### 日志

```bash
export CANNBOTDSL_LOG_LEVEL=10        # logging 级别数值，10 = DEBUG
export CANNBOTDSL_LOG_TO_CONSOLE=1
export CANNBOTDSL_LOG_TO_FILE=1
export CANNBOTDSL_LOG_FILE=/tmp/cannbotdsl.log
```

::: danger `LOG_*` 必须在 `import cannbotdsl` 之前设置
日志家族在**模块 import 时一次性**读取，进程内改了不生效。同理还有 `CANNBOTDSL_CACHE_MEM_MAX` 和 `CANNBOTDSL_CACHE_TRACE`。其余 pipeline / sync 类变量是**每次编译重读**，下一次编译就生效。
:::

### 编译耗时

```bash
export CANNBOTDSL_JIT_TIME_PROFILING=1
```

输出各阶段耗时。排查「首次调用为什么这么慢」时先看这个，再决定要不要上 AOT。

## 影响同步与正确性的开关

这组比转储更要紧：它们**改变生成的同步代码**，用错会从「慢」变成「错」。

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `CANNBOTDSL_AUTO_BUFFER_SYNC` | **开** | Buffer / Channel 的自动同步。为同核 Buffer 的跨 pipe 读写插入 `get_buf` / `release_buf` 并补齐读写完成依赖 |
| `CANNBOTDSL_INTRABLOCK_MATCH_MANUAL` | 关 | 为显式整数 ID 的核内同步补齐函数入口 priming 与出口 drain |
| `CANNBOTDSL_BUFFER_READ_SHARING` | 开 | 读共享优化 |
| `CANNBOTDSL_DEBUG_ASC_SYNC` | 关 | 取 `pipe` / `vf` / `all`，插入对应范围的同步调试操作 |
| `CANNBOTDSL_BARALL` | 关 | 给毕昇追加 `-mllvm -cce-aicore-sync-bar-all=true`，打开全局 `bar.all` 自动插入 |

::: danger 不要用 `AUTO_BUFFER_SYNC=0` 来「兼容」老代码
显式 `produce()` / `consume()` 选槽**必须**开启自动同步。`AUTO_BUFFER_SYNC=0` 只用于全手工同步的路径，不是 Channel 接口的兼容开关。关掉它再用 Channel，结果是数据错而不是报错。
:::

`CANNBOTDSL_BARALL=1` 是定位同步问题的大锤：全局插屏障，正确性优先、流水并行性全丢。**它不能替代跨核握手**——确认「加了 bar.all 就对了」只说明缺同步，还要回去找到底缺在哪里，见[同步、Cache 与跨核交接](./synchronization.md)。

::: tip 一个容易漏的同步点
同一个 Buffer 上连续两次 MTE2 纯写（包括 `fill_l1` / GM→L1，以及零次循环前后的 GM→UB），如果要求后一次覆盖前一次，需要显式调用 `cube_sync_pipe(PIPE.MTE2)` 或 `vec_sync_pipe(PIPE.MTE2)`。同一 pipe 上的 Buffer 锁**不能**代替这个完成屏障。
:::

## 会静默改变数值结果的开关

### `--cce-ftz`：非规格化数的处理

NPU ARCH 3510 的硬件**不实现非规格化数**（subnormal）。基础接口对它的支持是编译器软件模拟的，由毕昇编译选项 `--cce-ftz` 控制，而它**默认为 `true`**，即默认 flush-to-zero。

后果：当中间结果落进非规格化区间时，设备上得到 0，CPU 上的 golden 得到一个极小的非零值。

::: danger 这是一类很容易误判的精度问题
症状是**个别元素误差极大**，而不是整体误差偏大。很容易被当成掩码越界或搬运方向错。排查精度时先确认是不是 FTZ 导致的，再去查搬运。完整的排查顺序见[调试与精度验证](./debugging.md#精度不对时的四个嫌疑人)。
:::

### 低精度模式开关

这些开关一旦打开就改变整个 Kernel 的 Cube 行为，属于「影响代码结构的配置」，应该放在 `@kernel` 类的 `__init__` 里由编译期常量控制。`cannbotdsl.ops.cube` 公开的全部开关只有这些：

| 接口 | 作用 |
| --- | --- |
| [`enable_hf32`](../api/compute/matrix/enable-hf32.md) / [`set_hf32_round_mode`](../api/compute/matrix/set-hf32-round-mode.md) | FP32 降精度换吞吐；舍入模式用 `HF32RoundingMode` |
| `set_fp32_mode` | FP32 计算模式 |
| [`enable_fp8`](../api/compute/matrix/enable-fp8.md) | FP8 矩阵乘 |
| [`enable_hif8`](../api/compute/matrix/enable-hif8.md) | HiF8 矩阵乘 |
| [`set_mmad_direction`](../api/compute/matrix/set-mmad-direction.md) | 取 `"m"` / `"n"`；开 UnitFlag 时必须与搬出读取顺序一致 |

::: warning 上面这张表就是全部，没有别的
`cannbotdsl.ops.cube` 的 `__all__` 只有 `HF32RoundingMode` 加上表里这六个函数。

**别去找 `enable_hf32_trans`。** 有些接口清单里还能见到这个名字，但 0.7.0 的包里没有它，包内测试也把那条路径标为 obsolete。转置由 `mem_copy(..., transpose=True)` 负责，和 HF32 开关是两件事。
:::

### 目标架构

AOT 构建的默认目标是 `dav-3510`，可用 `--target` 切换；运行时也可以用 `CANNBOTDSL_NPU_ARCH` 覆盖探测结果（优先级：env > 运行时 SoC 查询 > 默认 `dav-3510`），跨编译场景用它保证确定性。

```bash
./run-build.sh --group ds41 --target dav-3510
```

设备侧要按架构分支时，用 `target_version()` 把**调用方算好的**布尔值标记成编译期条件：

```python
IS_DAV_3510 = cbd.get_platform_info().npu_arch == "dav-3510"

@cbd.kernel
def k(out):
    if cbd.target_version(IS_DAV_3510):
        ...
```

注意 `target_version()` **自己不查硬件版本**，只负责把一个已算好的布尔值标成编译期条件。

## 编译缓存

缓存是**两级**的，而且磁盘级可以跨进程复用。

| 级别 | 开关 | 行为 |
| --- | --- | --- |
| L1 内存 LRU | 默认开。容量 `CANNBOTDSL_CACHE_MEM_MAX`，默认 **128**；`0` 禁写，`none` 不限 | 进程内复用 |
| L2 磁盘 | **设置 `CANNBOTDSL_CACHE_DIR` 即启用** | 跨进程复用，布局 `$DIR/v2/<key 前两位>/<key>.so` 加一个 sidecar |
| 临时关闭 | `CANNBOTDSL_CACHE_MEM_DISABLE` / `CANNBOTDSL_CACHE_DISK_DISABLE` | 已有条目保留，重新开启后仍可复用 |
| 人工作废 | `CANNBOTDSL_CACHE_FORMAT_TAG` | 改这个值就让旧条目全部失效 |

缓存键由这几节内容拼成：`verified-ir`、`call-contract`、`environment`、`compile-options`、`target`、`runtime`、`toolchain`、`format`、`user-format`。原则是**黑名单制**——新增的环境字段默认进 key，只有能证明「不影响产物」时才排除。理由是不对称的：*stale miss 只是慢，stale hit 是正确性 bug*。

::: tip 冷启动时间有两条路
**轻量**：部署镜像里预置 `CANNBOTDSL_CACHE_DIR`，构建阶段跑一遍预热脚本把 `.so` 烤进镜像。改造成本低，但仍要命中键才有效，而键里含工具链与环境——工具链一升级就全失效。

**彻底**：[AOT 与 Native 算子包](./aot-packaging.md)。配置要事先枚举，但产物可审计、可复现。

要求严格可复现的线上服务走 AOT；内部服务和 CI 用磁盘缓存通常就够了。
:::

Host 侧的 API：

| 手段 | 作用 |
| --- | --- |
| `cbd.compile(host_fn, *specs)` | 显式提前编译，返回可复用的程序对象 |
| `cbd.clear_compile_cache()` | 清空进程内 L1 |
| `cbd.clear_compile_cache(clear_disk=True)` | 连磁盘 L2 一起清 |
| `cbd.clear_compile_cache(reap_orphans=True, orphan_min_age_seconds=3600.0)` | 回收孤儿产物 |

完整签名是 `clear_compile_cache(*, clear_disk=False, reap_orphans=False, orphan_min_age_seconds=3600.0)`，全部为关键字参数。

## AOT 与 Native 包相关的变量

容易混淆的一点：下面大部分是**打包工程 `net/native_package` 的构建脚本**读的，不是 DSL 运行时读的。两类分开列：

| 变量 | 谁读 | 作用 |
| --- | --- | --- |
| `CANNBOTDSL_NATIVE_BINARY_MODE=require` | **DSL 运行时** | 未命中预编译二进制时立即报错，不回退 JIT。上线前自检用 |
| `CANNBOTDSL_NATIVE_BINARY_REPORT` | **DSL 运行时** | 把预编译二进制的命中记录写成 JSON |
| `CANNBOTDSL_WHEEL` | 构建脚本 | 指定打包用的 DSL wheel |
| `CANNBOTDSL_ROOT` | 构建脚本 | 指定仓库根 |
| `CANNBOTDSL_DS41_PROFILES` | 构建脚本 | 用外置 JSON 覆盖默认导出配置集合 |
| `CANN_ENV` | 构建脚本 | 指定 `set_env.sh` |

用法与产物布局见 [AOT 与 Native 算子包](./aot-packaging.md)。

## 变量的读取时机

改了变量没生效，先看它属于哪一类：

| 时机 | 变量 | 不生效时怎么办 |
| --- | --- | --- |
| 每次编译重读 | pipeline / sync 两组，以及 `CACHE_DIR`、`CACHE_MEM_DISABLE`、`CACHE_DISK_DISABLE` | 下一次编译自然生效；同一次编译固定用入口处的快照 |
| **import 时一次** | `CANNBOTDSL_LOG_*`、`CACHE_MEM_MAX`、`CACHE_TRACE` | 必须在 `import cannbotdsl` **之前**设置 |
| 进程内首次计算后带 stat 戳记 | `ASCEND_HOME_PATH`、`CANNBOTDSL_BISHENG_PATH` | 换工具链要重启进程；同一文件被改写时按 `(mtime, size)` 自动重算 |

布尔取值：未设置、空串、`0`、`false`、`False` 为假，其余为真。缓存开关更严格且大小写无关：`1/true/on/yes` 为真，`0/false/off/no/空串/未设置` 为假，其他值记 WARNING 并按假处理。`CANNBOTDSL_FRONTEND_LINE_INFO` 和 `CANNBOTDSL_BARALL` 是二进制开关，显式非空值只接受 `0` 或 `1`，其他值直接报错。`CANNBOTDSL_LOG_LEVEL` 取整数，不是布尔。

::: warning 这些是包内变量，不是文档站承诺的接口
上面大部分变量在官方 API 文档站上没有对应页面。它们确实在 `cannbotdsl` 包里（`cannbotdsl/core/utils/env.py` 集中管理），可以放心用于**调试**；但不要写进生产脚本的硬依赖，版本间可能变化。
:::

## 运行期的三层观察

编译期的产物能看到之后，运行期还有三层证据。

### 第一层：构图期 `print` —— 验证「生成了什么样的 kernel」

最便宜且最有效。Python 内建 `print` 在构图期执行，看到的是真实的 Python 值：

```python
@kernel
class MyKernel:
    def __init__(self, num_col, dtype):
        self._tile = compute_tile(num_col, dtype)
        self._branch = pick_branch(num_col)
        print(f"[构图] num_col={num_col} tile={self._tile} branch={self._branch} "
              f"seg={self._num_seg} depth={self._depth}")
```

能在构图期发现的问题不要留到设备上：tile 算成 0 或负数、分支选错、循环次数不对、UB 预算算超。**把关键配置全部打出来**是成本最低的「看生成了什么」。

### 第二层：设备侧打印与转储 —— 验证「算出来对不对」

有两套互补的工具：

| 工具 | 看什么 | 在哪调用 |
| --- | --- | --- |
| `cbd.print(value, label=..., mode=...)` | UB / L1 / L0C 上的 Tensor，以及局部标量和点索引标量 | `@kernel` 内 |
| `cbd.dump_tensor(tensor, desc=..., dump_size=..., offset=..., shape=..., dtype=...)` | Tensor 的原始字节，走 CANN native dump 流 | `@kernel` 内 |
| `cbd.dump_reg(register, desc=..., dump_size=..., dtype=...)` | **矢量寄存器的 lane 内容** | 必须在 `with vf(mode="simd")` 内，对 `RawVReg` 调用 |

::: tip `dump_reg` 正好补上 `cbd.print` 的盲区
`cbd.print` 只支持 UB / L1 / L0C，看不到寄存器。而 Reg 编程模型的全部价值就是「中间结果留在寄存器里」——最想看的恰恰是 `cbd.print` 看不到的那一段。`dump_reg` 填的就是这个洞：它直接读 `RawVReg` 的 lane，调 VF 时很有用。

代价：走的是 CANN native dump 流，不出现在 `get_debug_prints()` 的结构化记录里，单次 VF 的载荷区满了之后会丢弃后续记录。
:::

`cbd.print` 的约束要记住：

- 只支持 **UB、L1、L0C** 上的 Tensor，以及 Kernel 局部标量和点索引标量。
- Channel 不能直接打印，先 `produce()` / `consume()` 取出 Tensor。
- 按**物理存储顺序**采样，最多 1024 字节。NZ / ZN 数据不会重排成 ND，看到的是分形顺序。
- 每个打印点在每个 block / subblock 上保留 **64 条**记录（`mode="ring"` 留最近的，`"stop"` 留最早的）。
- **会增加设备内存占用并触发 Host 回读与同步，不能用于性能测量。**

Host 侧还能拿到结构化记录，用于**自动比对**而不是肉眼看 stdout：

```python
from cannbotdsl.core.diag.debug import clear_debug_prints, get_debug_prints

clear_debug_prints()
run(...)
torch.npu.synchronize()
for r in get_debug_prints():
    if r["label"] == "stage1_out":
        compare_with_reference(r["shape"], r["data"])
```

每条记录含 `label`、`dtype`、`shape`、`physical_shape`、`memloc`、`physical_format`、`block_id`、`subblock_id`、`counter`、`data` 等字段。典型用法：在 kernel 的每个阶段末尾打一个带标签的点，Host 侧逐阶段与 numpy / torch 参考实现比对，定位**第一个**出错的阶段。

### 第三层：msprof —— 验证「行为是否符合预期」

- **Task Duration 与理论上限的比值** → 瓶颈在哪一侧。
- **各流水的占用与空档** → 流水是否真的重叠了，`depth` 够不够。
- **同一份代码改了 `unroll` 后 duration 没变** → 这个值可能没进设备代码；现在可以直接 `DUMP_ASCENDC` 去 `.asc` 里确认，不用再靠猜。

仓库样例的性能数据统一用 msprof 采集 Task Duration，每个 shape 重复多次取最小值。设备侧还可以用 [`get_system_cycle()`](../api/execution/device/get-system-cycle.md) 读周期数做粗粒度计时——但它会引入标量指令，别放在最内层循环里。

## 一个可照搬的验证流程

```text
① 构图期 print 全部编译期配置        → 确认生成的是你想要的那份 kernel
② CANNBOTDSL_DUMP_ASCENDC=1 看 .asc  → 确认循环、同步、常量真的是那样
③ cbd.print / dump_reg 逐阶段打标签  → Host 侧与参考实现比对，定位第一个出错阶段
④ clear_compile_cache() + 参数化测试 → 覆盖整除 / 尾块 / 单 tile / 超大四类边界
⑤ 关掉所有 dump 和 print，msprof 采集 → 和理论上限比
⑥ 改一个变量，重跑 ④ 和 ⑤
```

第 ⑤ 步前必须关掉 `cbd.print` 和所有 `DUMP_*`：前者会触发回读与同步，后者会旁路缓存强制重编译。

## 真正还缺的能力

把上面这些列完之后，确实拿不到的只剩这些：

| 能力 | 状态 |
| --- | --- |
| 编译优化等级选择（O0～O3） | **无开关**，毕昇固定 `-O3` |
| 编译诊断等级与分类开关 | 无 |
| 错误码表（`DiagnosticCode` 的取值与含义） | 未公开取值表，仅在 [Host API](../api/compilation/host/index.md) 提及类型名 |
| 源码行号关联到 Python 行 | `CANNBOTDSL_FRONTEND_LINE_INFO` 有，但语义未公开 |
| 汇编（而非 AscendC 源码）转储 | 无直接开关；`.asc` 之后交给毕昇，要看汇编得自己用毕昇工具链 |

::: info 这张表是全书唯一的缺口清单
[当前限制、迁移与常见问题](./limitations.md)一章引用它，不另立一张会脱节的表。
:::

## 下一步

[高性能算子编写指南](./performance.md)：产物能看见了，接下来是把它调快。
