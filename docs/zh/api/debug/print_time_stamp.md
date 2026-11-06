# print_time_stamp

## 产品支持情况

| 产品 | 已验证型号 | 支持情况 |
| --- | --- | --- |
| Ascend 950PR&950DT系列产品 | Ascend 950PR | 支持 |

支持状态限于表中已验证型号。验证环境为CANN 9.2、dav-3510、x86_64 Host、Python 3.12和TorchNPU 2.9.0.post8。下文AIC表示Cube计算核，AIV表示Vector计算核，MIX表示二者协同执行的混合核；VF（Vector Function）表示矢量函数。相关概念见[术语表](../../guide/glossary.md)。

### 已验证的支持范围

以下范围指本DSL实现的验证结果。AscendC原生接口的产品支持列表不能直接作为本实现已完成适配和验证的依据。

| 场景 | 支持范围及验证边界 |
| --- | --- |
| AIC、AIV标量代码 | 支持在 `@kernel` 的标量部分打点，已验证单block和4-block执行。 |
| MIX标量代码 | 支持分别输出AIC、AIV核信息，已验证4个AIC和8个AIV参与执行。这里的“多核”均在一张NPU卡内。 |
| 在线编译、编译产物保存加载 | 已验证 `cb.compile()` 后直接运行，以及 `program.save()`、`cb.load()` 后运行。保存的产物必须已经包含打点。 |
| ID、循环及重复执行 | 支持uint32范围内的Python整数常量和 `dtypes.uint32` 动态标量；已验证边界值、重复ID、循环和多次kernel提交。 |
| 打屏和二进制落盘 | 已验证AIC、AIV、MIX打屏，以及加载 `dump.json` 后保留打屏并生成按核区分的 `.bin` 文件。 |
| 官方工具解析 | 已验证 `show_kernel_debug_data` 解析为按核目录组织的CSV；解析后的ID、周期及程序计数器（Program Counter，PC）等字段与打屏核对一致。 |

### 明确不支持的范围

| 场景 | 当前行为或限制 |
| --- | --- |
| 普通Python或 `@host` 内直接调用 | 明确报错；本接口只能在设备核函数上下文中生成打点。 |
| VF函数体内调用 | SIMD VF和SIMT VF均明确报错。应在AIC/AIV核的标量部分、VF调用之外安排打点。 |
| 纯SIMT kernel | 代码生成阶段明确报错；即使打点写在 `with cb.vf(mode="simt")` 外，也不支持。 |
| CPU仿真输出 | 不支持；本接口使用NPU原生诊断通道。CPU上的代码生成或参数校验测试不代表CPU仿真能打印时间戳。 |
| Graph / 算子入图打印 | 按 [AscendC原生接口约束](https://asc.gitcode.com/api/Utils-API/tuning_interface/asc_time_stamp.html)，暂不支持。DSL不保证在capture/replay场景中输出；该限制不等同于一定会在编译阶段报错。 |
| 非法ID | 不接受负数、超过 `2**32-1` 的常量、布尔值、浮点值或非 `dtypes.uint32` 的动态标量。具体检查阶段见“参数说明”。 |
| 给旧算子包在运行时补打点 | 不支持；修改环境变量或加载Dump配置不能为未包含打点的二进制添加打点，必须修改算子并重新编译。 |
| 自动源码定位、异常报告或全流水计时 | 本接口不生成源码行号映射、AIC error报告，也不会自动等待所有计算流水完成。`descId` 由用户关联源码位置。 |

## 功能说明

在设备核函数的标量代码位置记录打点编号、设备周期计数和程序计数器（PC），并补充核类型、核索引和逻辑block信息，用于分析同一核执行到各个代码位置的时间。日志由CANN异步输出，也可按[原生二进制落盘与解析](#原生二进制落盘与解析)保存为文件；接口不会自动等待全部计算流水完成。

## 函数原型

```python
print_time_stamp(marker_id: int) -> None
```

从`cannbotdsl`导入；支持位置参数和`marker_id=`关键字参数。动态标量的DSL类型要求见下表。

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `marker_id` | 输入 | 范围为 `0` 至 `2**32-1` 的Python整数常量，或类型为 `dtypes.uint32` 的DSL动态标量。建议使用大于 `0xffff` 的ID，避免与AscendC内部保留ID冲突。 |

`marker_id` 表示用户指定的**打点位置编号**；Python调用、IR操作数和生成的DSL封装统一使用该名称，对应的原生日志字段仍为 `descId`。建议给不同源码位置分配不同编号；它不是核编号、设备编号或自动生成的源码行号。

布尔值、浮点值和越界常量在编译阶段报错。动态ID仅接受 `dtypes.uint32`，`int32`、`int64`、`uint64` 等其他类型在编译阶段报错；不再将任意整数隐式转换为打点ID。相同ID的连续调用和循环中的调用都会保留。

编译时将ID参数指定为 `dtypes.uint32` 后，Python调用传入的值由框架已有的uint32参数绑定检查范围，非法值在下发前报错。核函数内若需要转换其他整数表达式，应由调用者确认范围后显式使用 `dtypes.uint32(value)`；显式转换沿用DSL的整数转换语义。

## 返回值说明

无返回值。信息由CANN原生通道异步打印，不返回Python对象，不进入 `get_debug_prints()`，也不生成 `.pt` 文件、AIC error JSON报告或源码转储文件。运行时加载CANN Dump配置后，还可保存原生二进制记录，并用官方工具离线解析为CSV，见下文“原生二进制落盘与解析”。

## 约束说明

- 在 `@kernel` 的标量部分调用，也可在由该核函数调用的 `@jit` 辅助函数中调用。不能在普通Python、`@host`、任何VF区域或纯SIMT核函数中使用；这些场景会明确报错。
- 当前接口面向NPU真机调试，不支持CPU仿真。按原生接口限制，暂不支持算子入图场景的打印。
- 输出使用有限容量的原生调试缓冲，与其他原生转储功能共享容量；缓冲耗尽时记录可能缺失。避免在高频循环中无限量打印。
- 打点本身包含诊断写入，会改变算子耗时。补充核信息会额外写入一条原生printf记录，增加调试开销及缓冲占用。两个打点间的差值包含打点引入的成本，不能直接当作无打点版本的性能。未调用本接口的模块不会生成该辅助函数。timestamp编译宏开启时，即使 `ASCENDC_DUMP=0`，核信息仍随时间戳输出。
- 接口不会自动插入用于测量的全流水同步。标量执行到打点不表示此前异步MTE、Vector或Cube操作全部完成。测量完成时间时，需要按算子原有同步协议安排打点。
- 收集日志前同步对应设备或流，并允许CANN异步日志完成输出。强制终止进程、设备故障或缓冲耗尽时，不保证所有记录打印完成。

## 调用示例

下面的完整示例在AIC核上写入结果，使用固定编号 `71001` 和 `71002` 标记两个源码位置，无需通过host传入打点编号。它演示接口使用，不代表对写操作完成时长的精确测量。

```python
import torch
import torch_npu
import cannbotdsl as cb
from cannbotdsl import Tensor, TensorSpec, dtypes, scalar
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel


@kernel
def marked_kernel(output: Tensor):
    cb.print_time_stamp(71001)
    scalar.cube_store_bypass(output.ptr(0), 17)
    cb.print_time_stamp(71002)


@host
def run(output: Tensor):
    marked_kernel[1](output)


def main():
    program = cb.compile(run, TensorSpec((1,), dtypes.int64))
    try:
        output = torch.zeros(1, dtype=torch.int64, device="npu")
        program(output)
        torch.npu.synchronize()
        assert output.cpu().item() == 17
    finally:
        program.close()
    print("timestamp example passed")


if __name__ == "__main__":
    main()
```

在已配置CANN和对应框架包的环境中，将示例保存为 `timestamp_demo.py` 并执行：

```bash
python timestamp_demo.py > timestamp.log 2>&1
```

预期结果：

```text
timestamp example passed
```

同时可以看到内部入口记录以及 `descId is 71001` 和 `descId is 71002`。若缺少输出，先确认实际运行的是包含打点的新编译产物，再检查是否进入了相应分支、是否处于入图场景、ID是否合法，以及调试缓冲是否已被其他转储操作占满。

## 日志字段与多核分析

包含打点的编译模块自动定义 `ASCENDC_TIME_STAMP_ON` 并引入对应SDK头文件，无需设置环境变量。未含打点的模块不会添加这些宏、头文件或打点调用。已编译的算子包不会因运行时配置而新增打点：添加或移除调用后需要重新编译算子。

输出分为原生时间戳行和补充核信息行，格式如下（动态值使用占位符）。实际日志中两类行可能与其他核的输出交错，不能依赖它们相邻：

```text
descId is 71001, rsv is 0, timeStamp is <cycle>, pcPtr is <pc>, entry is <entry>.
[AIC Block <block>/<count>] [CANNBotDSL][timestamp-core] coreType=AIC coreIndex=<core> blockIdx=<block> descId=71001 entry=<entry>
```

| 字段 | 所在日志 | 含义 |
| --- | --- | --- |
| `coreType` | 核信息行 | 执行该打点的核类型：`AIC` 或 `AIV`。MIX会分别产生这两类记录。 |
| `coreIndex` | 核信息行 | CANN原生诊断缓冲使用的核索引，取自SDK的 `asc_debug_get_core_idx()`。不是设备编号，也不保证等于逻辑block索引；不要把它当成原始硬件 `get_coreid()` 值。 |
| `blockIdx` | 核信息行 | CANN调试通道中的逻辑block索引。AIC使用block编号；AIV按 `block * subblock_dim + subblock_id` 展开，因而可以区分MIX中两个AIV子核。 |
| `descId` | 两类行 | 用户传入的打点ID，仅用于识别代码位置，不编码核类型或核索引。不同核执行同一位置时ID相同。 |
| `timeStamp` | 原生时间戳行 | Scalar执行到打点时采样的设备系统计数器值，单位为计数周期（cycle），不是主机时间或Unix时间戳。 |
| `pcPtr` | 原生时间戳行 | SDK打点实现内部采样的PC，原生日志以十进制显示。可用Python的 `hex(int(value))` 转换为十六进制。 |
| `entry` | 两类行 | 本次kernel在当前核上开始执行时的设备周期计数，不是当前打点函数或普通子函数的入口时间。 |

SDK还会输出内部入口记录，通常为 `descId=0`。同一核的连续打点可计算 `timeStamp` 差值；换算为时间需要使用对应芯片的计数频率。MIX核会由AIC和AIV各自执行打点，因此一个源码调用可能输出多条记录。日志交付顺序不等同于跨核执行顺序，不能直接将相邻两行当作同一核的开始和结束。

筛选 `[CANNBotDSL][timestamp-core]` 可看到各核执行了哪些打点。按当前进程、当前kernel调用的 `(coreType, coreIndex, blockIdx, entry)` 分组，再用 `descId` 识别打点位置。每次用户打点补充一条核信息行；SDK自动生成的内部入口记录不额外补充核信息。`[AIV Block 3/8]` 表示该AIV逻辑block索引为3、总数为8，并不是物理3号卡。

核信息行不重新采样时间或PC。`entry` 也不是全局唯一的任务ID，不同核可能具有相同的 `entry` 和 `descId`，因此不能只凭这两个字段或日志相邻顺序将两类行唯一配对。需要精确确认时间戳和PC属于哪个核时，使用下文按核保存的原生 `.bin` 文件及解析结果。

原生时间戳行和核信息行来自同一次用户调用，不能将两者合计为两次打点。SDK的保留字段 `rsv` 不用于编码用户ID或核类型。两种记录分别写入有限缓冲，缓冲耗尽或进程异常退出时，不保证两份输出都完整；看到核信息行不代表对应原生时间戳一定已经成功输出。

`pcPtr` 不是用户DSL源码位置的唯一标识。SDK使用不可内联的打点函数，同一核上不同源码位置可能输出相同PC。请用 `descId` 关联代码位置；本接口不自动生成Python或AscendC行号映射，也不依赖AIC error异常报告。

### 多核场景下的entry和耗时

`entry` 取自 `__get_entry_sys_cnt()`，表示设备侧的kernel执行起点，不包含host提交后的排队时间。进入普通子函数或再次调用打点函数不会重新设置这个起点。同一核、同一次执行中的多个打点共享 `entry`；不同核不保证同时开始执行，因此 `entry` 不保证相同。

下面是单卡4个AIC核执行同一kernel时的一组实际日志数据：

| `coreIndex` | `blockIdx` | `entry` | 相对核索引0的入口差值 |
| --- | --- | --- | --- |
| 0 | 0 | 4918641225655 | 0 cycles |
| 1 | 1 | 4918641225684 | 29 cycles |
| 54 | 2 | 4918641225854 | 199 cycles |
| 55 | 3 | 4918641225887 | 232 cycles |

这些核索引属于同一张NPU卡，不是设备编号。MIX中AIC、AIV的 `entry` 也可能不同。分析耗时时：

- 从某个核的入口到打点位置，使用该条记录的 `timeStamp - entry`。
- 从一个打点到另一个打点，使用同一核、同一次执行的两个 `timeStamp` 相减。
- 差值单位为系统计数周期；换算为微秒需要对应芯片的系统计数频率，不能直接使用计算核的工作频率。
- 不按日志出现顺序跨核配对，也不只凭 `entry` 相同与否判断是否属于同一次kernel调用。

## 原生二进制落盘与解析

以下示例演示普通Python调用AIC kernel后保存和解析时间戳。相同流程已在Ascend 950PR、CANN 9.2、TorchNPU 2.9.0.post8上完成AIC、AIV和MIX验证：加载Dump配置后生成原生 `.bin` 文件，解析得到的ID、cycle和PC与终端日志一致，同时保留打屏输出。

编译宏 `ASCENDC_TIME_STAMP_ON` 控制算子是否包含打点记录，CANN Dump配置控制运行时是否保存记录。前者由DSL自动添加；后者需要启动脚本显式加载。只创建 `dump.json` 不会自动启用落盘，也不会给未包含打点的算子补充打点。

### 准备配置文件

创建一个独立的工作目录，并在该目录中保存下面的 `dump.json` 和 `timestamp_dump_demo.py`：

```bash
mkdir -p "$HOME/timestamp-demo/timestamp-output"
cd "$HOME/timestamp-demo"
```

`dump.json` 内容如下：

```json
{
  "dump": {
    "dump_kernel_data": "timestamp",
    "dump_path": "./timestamp-output"
  }
}
```

- `dump_kernel_data` 指定保存时间戳记录。
- `dump_path` 指定数据保存目录，应提前创建并确保可写。相对路径按进程的当前工作目录解析；也可以改成绝对路径。
- 配置文件名称和位置可以自行选择，由 `set_dump()` 的路径参数指定。示例从Python脚本所在目录读取配置。
- CANN的保存路径优先级为 `ASCEND_DUMP_PATH` > `ASCEND_WORK_PATH` > JSON中的 `dump_path`。若设置了前两个环境变量，应到对应目录检查结果。本路径不由 `CANNBOTDSL_DUMP_DIR` 控制。

配置字段详见 [aclmdlSetDump接口说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_0315.html)。

### 在启动脚本中加载配置并运行算子

将下面的完整代码保存为 `timestamp_dump_demo.py`。`init_dump()`、`set_dump()` 和 `finalize_dump()` 在普通Python启动流程中调用，不放在 `@kernel` 或 `@host` 内部，也不必为每次kernel提交重新调用。

```python
from pathlib import Path

import torch
import torch_npu
import cannbotdsl as cb
from cannbotdsl import Tensor, TensorSpec, dtypes, scalar
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel


@kernel
def marked_kernel(output: Tensor):
    cb.print_time_stamp(71001)
    scalar.cube_store_bypass(output.ptr(0), 17)
    cb.print_time_stamp(71002)


@host
def run(output: Tensor):
    marked_kernel[1](output)


def main():
    config_file = Path(__file__).resolve().with_name("dump.json")
    # 0是进程可见设备编号；请沿用业务原有的设备选择逻辑。
    torch.npu.set_device(0)
    program = None
    torch_npu.npu.init_dump()
    try:
        # 在加载、执行算子之前配置Dump。
        torch_npu.npu.set_dump(str(config_file))
        program = cb.compile(run, TensorSpec((1,), dtypes.int64))
        output = torch.zeros(1, dtype=torch.int64, device="npu:0")
        program(output)
        torch.npu.synchronize()
        assert output.cpu().item() == 17
    finally:
        try:
            if program is not None:
                program.close()
        finally:
            torch_npu.npu.finalize_dump()


if __name__ == "__main__":
    main()
```

在已安装当前DSL框架和配套TorchNPU的Python环境中执行。以下CANN路径以验证环境为例，请按实际安装位置调整；设备可见范围应按分配的设备配置，示例中的 `npu:0` 不一定对应物理0号卡。

```bash
source /usr/local/Ascend/cann-9.2.0/set_env.sh
cd "$HOME/timestamp-demo"
python timestamp_dump_demo.py
```

终端仍会输出 `descId is 71001`、`descId is 71002` 等记录；在未被环境变量覆盖路径的情况下，原生数据位于 `timestamp-output/时间目录/asc_kernel_data_aic_0.bin`。文件名和数量随核类型、核数及SDK版本变化，不应在业务逻辑中依赖固定文件名。

多核场景下，当前SDK按核生成 `asc_kernel_data_aic_<coreIndex>.bin` 或 `asc_kernel_data_aiv_<coreIndex>.bin`，原始文件位于同一个时间目录中。二进制文件头保存核类型和核索引，timestamp记录保存逻辑 `blockIdx`、`descId`、周期、PC和 `entry`。核身份不需要编码进 `descId`。

当前DSL复用TorchNPU的运行时初始化，无需在此流程中另外调用 `aclInit`。若使用已编译的算子包，将示例中的编译部分替换为加载逻辑，并在加载之前设置Dump；算子包本身需要已经包含打点代码。多进程任务应由各进程分别配置，使用独立输出目录，便于区分记录。

### 解析二进制文件

在同一个工作目录中执行：

```bash
show_kernel_debug_data ./timestamp-output ./parsed
```

第一个参数支持单个 `.bin` 文件或包含这些文件的目录，第二个参数是解析结果的保存目录。若原生数据被环境变量保存到其他位置，应相应替换第一个参数。

工具随CANN发布。若命令不存在，先检查CANN环境配置；若提示 `No module named show_kernel_debug_data`，需要在当前Python环境安装配套解析工具。验证环境中的安装包示例为：

```bash
python -m pip install /usr/local/Ascend/cann-9.2.0/tools/show_kernel_debug_data-0.1.0-py3-none-any.whl
```

安装路径和包版本应以实际CANN软件包为准。工具用法详见 [show_kernel_debug_data文档](https://asc.gitcode.com/guide/programming_guide/appendix/show_kernel_debug_data_tool.html)。

解析后可找到类似下面的文件：

```text
parsed/
└── PARSER_解析时间/
    ├── dump_data/
    │   ├── 0/
    │   │   └── time_stamp_core_0.csv
    │   └── 18/
    │       └── time_stamp_core_18.csv
    └── parser.log
```

这里以多核结果示意目录布局，实际核索引和文件数量取决于执行情况；单核示例不会生成所有这些目录。当前官方工具使用核索引作为目录名，不额外增加AIC/AIV目录层级。核类型可结合原始 `.bin` 文件名、文件头和带 `coreType` 的打屏记录核对。

### 阅读解析结果

以下是验证时AIC核的一组实际数据，具体数值会随执行变化：

| 打点标识 | Cycle | Cycle间隔 | PC指针 |
| --- | --- | --- | --- |
| TIME_STAMP_WRAP_FIRST | 7817693849148 | 7817693849148 | 19792505342748 |
| 71001 | 7817693849177 | 29 | 19792505344992 |
| 71002 | 7817693852873 | 3696 | 19792505344992 |

- **打点标识**对应传入 `print_time_stamp()` 的ID；`TIME_STAMP_WRAP_FIRST` 为SDK内部入口记录。
- **Cycle**是设备周期计数，不是主机时间。**Cycle间隔**是同一核相邻记录的计数差；上例中71001到71002的间隔为3696 cycles。首行没有前一条记录，其间隔不能作为执行耗时。
- **PC指针**是打点实现采样的地址，以十进制显示，不是源码行号。两个ID对应同一PC是可能的，应通过ID关联用户代码位置。

当前配套工具的CSV仅展示上述四列，未单独展示 `blockIdx`、`entry` 或核类型；这些字段仍保留在原始二进制和补充核信息行中。核索引由CSV所在目录及文件名区分。

时间换算公式为 `耗时（微秒）= Cycle间隔 / 计数频率（MHz）`，计数频率应按对应芯片确认。两个采样点的间隔包含打点本身的开销；涉及异步MTE、Vector或Cube操作时，还需结合原有同步位置解释结果。不同核的记录应分别分析，不能把不同CSV的相邻行直接当作一段kernel耗时。
