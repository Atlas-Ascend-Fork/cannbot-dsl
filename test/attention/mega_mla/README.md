# mega_mla 测试

原生 cache 用例复用本文件的 CPU/NPU 参考和 `_exercise_attention`：覆盖独立页 stride、
压缩侧每页 64/128 行、非零 storage offset、padding 哨兵、全量/索引解码、
多请求页表、packed/空请求、序列长度同址更新、寄存器/UB/分段页表路由，
以及同一 launcher 切换 stride 后再复用旧编译产物。

```bash
python3 -m pytest -q test/attention/mega_mla/test_mega_mla.py -k native_cache -m 'not npu and not ascendc_toolchain'
ASCEND_RT_VISIBLE_DEVICES=0 python3 -m pytest -q test/attention/mega_mla/test_mega_mla.py -k native_cache_device -m npu
```

算子接口和输入约束见[样例说明](../../../samples/attention/mega_mla/README.md)。
测试、独立 PyTorch 参考和性能采集统一位于
[`test_mega_mla.py`](test_mega_mla.py)，以下命令均从仓库根目录执行。

## 环境要求

主机侧测试需要 pytest 和PyTorch 2.7及以上（同时具有float8_e4m3fn与float8_e8m0fnu）。AOT 编译需要 CANN 与
CANNBotDSL 的 Layout 资源接口（`Buffer/RingBuffer(mem_loc, dtype, layout, ...)`）；
设备测试还需要配套 torch_npu 和 Ascend 950 NPU。发行包从
[PyPI 公开发行页](https://pypi.org/project/cannbot-dsl/) 获取，安装步骤见
[仓库说明中的安装章节](../../../README.md#2-安装-cannbot-dsl)。
设备测试默认使用当前流可用的 Cube/Vector 配对核数；`selected_grid` 用例显式覆盖少核、奇数核和非整除核数，
超过设备物理核数的设备用例会跳过。AOT 与 NPU 分组建议在独立进程执行。

## 精度测试

```bash
# 查看全部用例。
python3 -m pytest --collect-only -q test/attention/mega_mla

# 主机侧输入、长度、调度及 workspace 契约。
python3 -m pytest -q test/attention/mega_mla -m 'not npu and not ascendc_toolchain'

# CANN AOT 编译。
python3 -m pytest -q test/attention/mega_mla -m ascendc_toolchain

# 设备精度与重复调用。
ASCEND_RT_VISIBLE_DEVICES=0 python3 -m pytest -q test/attention/mega_mla -m npu
```

| 分组 | 主要覆盖 |
| :--- | :--- |
| 输入与资源契约 | shape/dtype、权重布局、索引与长度元数据、workspace 所有权、分核与缓冲容量 |
| 投影与 cache | MX scale、E4M3 NZ 权重及位模式准备、INT8/UINT8 载体迁移、投影尾块、FP8/FP4 记录、RoPE 与 cache 写入边界 |
| Attention | H8/H16/H64、小 token 与非整除分核、predecode/indexed 路径、重复调用 |
| 分页与变长请求 | 多请求页表、packed Query、空请求、无效索引、有效长度、宽页表 |
| Query 元数据 | 导出/跨层复用、输入原地更新、快照有效性与逐位一致性 |
| 函数入口 | shape 自动推导、独立输出、线程/设备/stream 缓存隔离、图捕获与重复调用 |

设备测试对照独立 CPU 数学参考，检查输出精度、cache/gather 字节及输入只读性；
具体数值容差与参数组合以测试代码为准。缺少依赖或设备时的跳过不代表设备精度通过。

按场景选择现有用例：

```bash
python3 -m pytest -q test/attention/mega_mla -k cache_record_layout -m npu
python3 -m pytest -q test/attention/mega_mla -k 'packed_query or general_page_table' -m npu
python3 -m pytest -q test/attention/mega_mla -k cross_layer_query_metadata -m npu
python3 -m pytest -q test/attention/mega_mla -k function_entry_device -m npu
python3 -m pytest -q test/attention/mega_mla -k selected_grid -m npu
```

## 性能测试

`benchmark` 支持普通执行与图重放；每次目标调用前（包括预热）对独立 FP32 buffer
读写并同步，默认缓存压力为 256 MiB。压力操作和 host 同步不计入目标 kernel 设备耗时。
首次编译和权重准备在测量循环之外，性能结果应同时记录输入规格与软件环境。

```bash
msprof --output=log/mega_mla --aic-metrics=PipeUtilization \
  --task-time=on --ai-core=on \
  --application="python3 test/attention/mega_mla/test_mega_mla.py benchmark --tokens 8 --heads 16 --k 640 --warmup 10 --repeat 50 --pressure-mib 256 --graph"
python3 test/attention/mega_mla/test_mega_mla.py analyze-profile \
  log/mega_mla --last 50
```

汇总脚本选取 mega kernel 的 `MIX_AIC` 任务，输出设备耗时及流水指标。比较不同实现时，
须使用相同输入、预热和缓存压力，并从统计中排除预热任务。
`benchmark` 测量基础 Attention 入口；需要测量带长度或元数据复用的接口时，应单独采集，
不能用基础入口的时间代替。

分页场景可使用 `benchmark-paged`。`--logical` 选择 kernel 内页表查找，省略时使用
预先计算的物理索引；所有参数可通过同一入口的 `--help` 查看：

```bash
python3 test/attention/mega_mla/test_mega_mla.py benchmark-paged \
  --source samples/attention/mega_mla --output log/mega-paged --logical
```

检视回归沿用同一测试文件，包含当前流AIC/AIV上限与查询失败、公开NZ存储校验、
无效L2阈值回退、T100尾块AOT/NPU、kernel内零行初始化以及测试线程数恢复。
需要设备或工具链的用例沿用对应测试分组，跳过不计作通过。
