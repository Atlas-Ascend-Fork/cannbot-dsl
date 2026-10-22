# HIF8 ReduceScatter

本样例用 CANNBot DSL 实现多卡 HIF8 ReduceScatter，支持 `sum`、`max`、`min`，以及 UBMem、URMA 数据传输。输入和输出使用 `torch.int8` 承载 HIF8 原始编码，设备端转换为 FP32 归约，最后一次转换回 HIF8。普通 int8 数值不能直接作为 HIF8 编码使用。

## 文件组织

- `reduce_scatter.py`：包含布局与分块策略、AIV 内核、编译入口和算子接口。
- `comm_utils.py`：内部通信辅助模块，复用外部 HCCL 通信组，准备链路、缓冲和私有 context；不创建或销毁通信组。
- `../../test/reduce_scatter/test_reduce_scatter.py`：pytest 布局、ABI、编译及多卡精度测试，包含离线编译与 torchrun worker。

历史性能扫描、调度器专用脚本和 URMA 诊断工具不属于算子接口。

## 语义与布局

设 rank 数为 P，每卡输出元素数为 N。默认输入包含 P 个长度为 N 的目标切片，每卡输出只保留自己的归约结果。以 sum 为例：

```text
out_rank[i] = sum(input_source[rank * N + i] for source in range(P))
```

每个来源将目标切片推送到目标卡通信窗口中自己的来源行；目标卡按输出位置分配 AIV，逐来源解码并进行 FP32 归约。数据窗口与通知窗口独立，单 bank 通过轮次编号和完成 barrier 安全复用。

默认每卡数据窗口预算 128 MiB，chunk 容量为 `buffer_bytes // (2 * P)` 向下对齐到 128 字节；归约 tile 默认 17664 个元素。所有 stride 参数以字节计，`count` 和 `tile_elements` 以元素计。

## 接口

从仓库根目录导入；模块导入需要安装 cannbot-dsl 和 opkit，布局计算无需初始化 NPU 或通信组：

```python
from samples.reduce_scatter.reduce_scatter import (
    AivReduceScatterSoleMesh, Layout, Plan,
)

# 调用方设置当前设备并初始化 torch.distributed，group 为已有的 torch HCCL 通信组。
plan = Plan(Layout(count=N, ranks=P, itemsize=1), num_blocks_limit=32)
with AivReduceScatterSoleMesh(
    plan, group=group, transport="urma", artifact_path="kernel.so",
) as op:
    op(x, out)  # 输入输出已分配，dtype=torch.int8，承载 HIF8 原始字节。
    # 若捕获了 graph，须在退出 with 前结束 replay 并 graph.reset()。
```

用户只提供 torch 通信组，不创建或传入 `RsMeshComm`。算子直接复用该组的 HCCL communicator，不调用 `dist.new_group()`，在内部准备通道和数据/通知缓冲，并将填好的结构体上传为通信辅助对象中的私有 `_context`。`auto` 协议探测也在内部完成。

所有 rank 同序创建/调用/销毁算子。`close()` 等待设备完成并释放私有 context 的引用，不销毁传入的组；HCCL 管理的窗口和通道随外部通信组销毁而回收。调用方必须在算子关闭、graph 退出后再销毁通信组。设备 kernel 和编译入口继续使用 `StructRef[RsMeshComm]`，这是内部参数，不是公开调用接口。

主要参数（布局与调度参数传给 `Layout` / `Plan`）：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `plan`（算子） | 必填 | 已构造的 `Plan`；其 `Layout` 显式描述卡数与输入输出布局 |
| `count`（Layout） | 必填 | 每个 rank 的输出元素数 |
| `ranks`（Layout） | 必填 | 通信组的 rank 数；由调用方提供 |
| `dtype` | `hifloat8` | 仅支持 HIF8；外部存储使用 int8 |
| `reduction` | `sum` | `sum` / `max` / `min` |
| `group` | `None` | 外部已初始化的 torch HCCL 通信组，默认 WORLD；内部直接复用 |
| `transport` | `ubmem` | `ubmem` / `urma` / `auto` |
| `num_blocks_limit` | 32 | 实际 block 数由切片大小决定 |
| `buffer_bytes` | 128 MiB | 每 rank 的数据窗口预算 |
| `tile_elements` | 17664 | 256～17664，必须是 32 的倍数 |
| `repeats` | 1 | 输入中独立的 ReduceScatter 数据组数 |
| `input_slice_stride` | `count` | 输入相邻目标切片的字节间隔 |
| `output_slice_stride` | `count` | 未指定 `output_repeat_stride` 时作为其默认值；单位为字节，必须不小于单次输出字节数 |
| `input_repeat_stride` | `input_slice_stride * P` | 相邻输入数据组的字节间隔 |
| `output_repeat_stride` | `output_slice_stride` | 相邻 repeat 输出起始地址的字节间隔；显式设置时覆盖默认值 |
| `artifact_path` | `None` | 已编译 `.so` 路径；未指定时编译当前 plan |

HIF8 每个元素占 1 字节，因此上述默认字节步长等于 `count`。每次输出的 `count` 个元素连续存放，`output_slice_stride` 不是输出元素之间的间隔。例如 `count=13`、`repeats=2`、`output_slice_stride=32` 时，两次输出分别从字节偏移 0 和 32 开始；若显式指定 `output_repeat_stride=64`，则从 0 和 64 开始。

算子不提供 `mode` 参数；普通执行与图捕获由调用方控制。使用 `torch.npu.graph(...)` 捕获调用，再使用 `graph.replay()` 重放，算子实例和通信上下文必须在重放期间保持有效。

输入输出必须连续、位于准备时的 NPU，且内存不重叠。算子检查设备及 stream 一致性，防止同一个对象被多个线程并发提交。调用异步入队，`close()` 等待设备完成后集体结束算子使用；关闭后禁止再次提交，关闭与提交不能并发。当前通信窗口和注册资源属于传入的 HCCL 组，同一组仅供一个存活算子实例使用；与其他 HCCL 操作必须串行执行。需要多实例或跨 stream 并发时，由调用方为各实例创建独立 HCCL 组。所有 rank 的 plan、transport 和调用顺序必须一致。

即使数据走 URMA，通知仍要求 peer 存在 UB_MEM 映射。测试脚本在外部初始化 HCCL 通信组并传入算子；纯 Gloo 组不支持此接口。

`artifact_path` 必须匹配当前源码、plan 和传输方式，且配套 `.meta.json` 同时存在。测试脚本按源码、配置和包版本生成产物名。源码、plan、传输方式或相关包版本变更后，应重新编译产物。

## 环境

已在 Ascend950 / `dav-3510`、CANN 9.2.0、cannbot-dsl/opkit 0.7.1 和配套 torch-npu 环境验证。运行前加载匹配的 CANN `set_env.sh`，安装仓库依赖；设备测试使用 torch-npu 的 HIF8 转换接口 `torch_npu.npu_dtype_cast`。其他硬件和软件组合需另行验证。

## 测试与 CI

仓库已在 `test/test_config.yaml` 与 `scripts/ci/operator_list.yaml` 注册 `reduce_scatter`。测试使用已有 `npu`、`cannir_install` 标记；多卡测试默认使用 2 张可见设备，每个参数用例启动独立 torchrun 进程。

```bash
# 不使用 NPU：布局、上下文 ABI、拓扑借用内存回归和 DSL 转换检查。
python -m pytest test/reduce_scatter -m 'not npu' -v

# 预约设备前离线编译 2 卡测试产物；目录由使用者指定。
python test/reduce_scatter/test_reduce_scatter.py \
    --prepare --ranks 2 --artifact-dir /tmp/rs-hif8-artifacts

# 已预约 2 张可见 NPU 后运行；显式设置产物目录可禁止测试阶段编译。
RS_DSL_TEST_RANKS=2 RS_DSL_ARTIFACT_DIR=/tmp/rs-hif8-artifacts \
    python -m pytest test/reduce_scatter -m npu -v

# 仓库统一入口也可使用：
bash scripts/ci/run_tests.sh --ops=reduce_scatter --mode=cpu
```

多卡测试覆盖两种传输、三种归约、非对齐尾部、多 chunk、小 tile、带 padding 的重复布局、输入不被修改、输出边界保护，以及更换输入后的图重放。参考结果先解码实际 HIF8 输入，按来源 rank 顺序做 CPU FP32 归约，再编码回 HIF8，逐字节比较。

不设置 `RS_DSL_ARTIFACT_DIR` 时，NPU pytest 会在启动设备 worker 前临时编译；需要严格分离离线编译和设备测试时使用上述两阶段命令。
