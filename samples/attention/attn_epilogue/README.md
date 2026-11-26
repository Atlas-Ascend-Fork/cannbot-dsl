# attn_epilogue

Ascend 950PR/950DT（ARCH 3510）上的本地 attention 输出投影：将 inverse RoPE、
两次 MXFP8 激活量化和两级 Matmul 融合为一个设备 kernel。接口和编译位于
[attn_epilogue.py](attn_epilogue.py)，切分位于 [attn_epilogue_tiling.py](attn_epilogue_tiling.py)，
设备流水位于 [attn_epilogue_kernel.py](attn_epilogue_kernel.py)。

## 接口

```python
from samples.attention.attn_epilogue.attn_epilogue import attn_epilogue

out = attn_epilogue(x, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos)
```

导入算子会注册 `torch.ops.cannbotdsl.attn_epilogue`，可用同一组参数直接调用。
支持任意 `T >= 1`，受设备内存限制；普通执行返回具有独立存储的 BF16 `[T,5120]`。
NG 表示本算子处理的本地分组数；算子只计算这些分组的输出投影，不执行跨卡通信或结果聚合。

| 场景 | 输入 x | 第二级 K |
| :--- | :--- | ---: |
| NG=8 | `[T,64,512]` | 8192 |
| NG=2 | `[T,16,512]` | 2048 |
| NG=1 | `[T,8,512]` | 1024 |

每组含 8 个 head，固定 `F=4096`、`O_LORA=1024`、`DIM=5120`。

| 参数 | ND 权重时的逻辑 shape | NZ 权重时的逻辑 shape | dtype |
| :--- | :--- | :--- | :--- |
| `x` | `[T,NG*8,512]` | 同左 | BF16 |
| `woa`（FP8） | `[NG,1024,4096]` | `[NG,4096,1024]` | FP8 E4M3FN |
| `woa`（FP4，按字节计） | — | `[NG,1024,2048]` | uint8 打包的 FP4 E2M1 |
| `wob` | `[5120,NG*1024]` | `[NG*1024,5120]` | FP8 E4M3FN |
| `descale_woa` | `[NG,1024,64,2]` | FP8 `[NG,64,1024,2]`，FP4 `[NG,1024,64,2]` | FP8 E8M0FNU |
| `descale_wob` | `[5120,NG*16,2]` | `[NG*16,5120,2]` | FP8 E8M0FNU |
| `rope_sin`、`rope_cos` | `[T,64]` | 同左 | FP32 |

双 FP8 权重同时使用 ND 或同时使用 NZ。FP8 NZ 使用标准 `FRACTAL_NZ`，权重按 `(K,N)` 排列。
FP4 `woa` 使用 `FRACTAL_NZ_C0_16`，搭配 FP8 NZ `wob`。激活和 scale 始终使用 ND，
FP4 的 `descale_woa` 为 `[NG,1024,64,2]`。所有输入须连续、位于同一 NPU，并符合表中的形状和类型。

入口通过 `torch_npu.get_npu_format` 识别权重布局。若从 uint8 文件加载 scale，
使用 `.view(torch.float8_e8m0fnu)` 恢复位模式。

## 计算与调度

inverse RoPE 仅处理每个 head 的最后 64 维，前 448 维保持原值。尾部按相邻元素配对，
在 FP32 中计算 `even'=even*cos+odd*sin`、`odd'=odd*cos-even*sin`，再舍入为 BF16。
sin/cos 表按 `[s0,s0,s1,s1,...]` 重复。MX 量化沿 K 每 32 个元素使用一个 E8M0 scale。

```text
A   = BF16(inverse_rope(x))
Y_g = BF16(MXMatmul(Q32(A_g), WoA_g.T))
out = BF16(MXMatmul(Q32(concat_groups(Y_g)), WoB.T))
```

MM1 在 FP32 中累加，并在 QuantY 前先舍入为 BF16。FP32 tile 通过 Fixpipe 直接交付配对
AIV，省去 BF16 中间 Y 的 GM 缓冲。内部只保留 Aq、A-scale、Yq、Y-scale 四个字节工作区。

实现使用 DSL 0.8.2 的显式 layout、RingBuffer 和 copy_options，并显式开启 MX Matmul。
Fixpipe 按完整 tile 的两个半帧交付 AIV，每个 AIV 独立裁剪有效行；空分片仍消费接收槽，
完成所有权交接，但不写入 GM。尾块不依赖按 T 或平台枚举的配置。

M/N/K tiling 根据 T、矩阵维度、当前 stream 核配额及 ARCH 3510 的 L0/L1/UB 容量推导，
没有按 T 或设备名称列举配置。MM1 的 N 分块对齐完整 MX scale 字节对；满足生产者覆盖条件时，
QuantA 和 MM1 按 K slab 流水。Cube 缓冲复用和 QuantY → MM2 的数据/scale 就绪依赖保留显式同步。

编译缓存按输入 shape、dtype、ND/NZ 布局、设备和当前 stream 的 AIC 核数复用已编译的句柄和工作区规格。
NCHW 等映射到同一 ND 规格的格式标记共用缓存键。
每次执行为四个工作区和输出分配张量，图捕获时使用图内存池。跨 stream 的依赖由调用方建立。
JIT 首次调用包含编译开销，可先预热。
当前提供前向实现，不提供反向。

## 权重准备

转换只在初始化时执行，不能放进被测 forward。按字节转换可避免部分 CANN 版本对 FP8/E8M0
设备转置的限制，且不会改变量化值。

```python
import torch
import torch_npu


def weight_to_nz(weight):
    """Preserve quantized bytes while converting logical (N,K) to standard NZ (K,N)."""
    kn = weight.view(torch.uint8).cpu().transpose(-2, -1).contiguous()
    return torch_npu.npu_format_cast(kn.npu().view(torch.float8_e4m3fn), 29)


wa_nz = weight_to_nz(wa_nd)
wb_nz = weight_to_nz(wb_nd)
sa_kn = sa_nd.view(torch.uint8).cpu().permute(0, 2, 1, 3).contiguous().npu()
sb_kn = sb_nd.view(torch.uint8).cpu().permute(1, 0, 2).contiguous().npu()
out = attn_epilogue(x, wa_nz, wb_nz,
    sa_kn.view(torch.float8_e8m0fnu), sb_kn.view(torch.float8_e8m0fnu), sin, cos)
```

FP4 `woa` 从行主序的 `[NG,1024,2048]` 权重转换，每字节含两个 FP4 值。
在 NPU 上使用 `uint8` 载体，官方 `npu_format_cast` 返回的实际格式为 `FRACTAL_NZ_C0_16`（50）。
WOA scale 使用原来的 ND 排布，WOB 及其 scale 按上面的 FP8 NZ 示例准备。

```python
wa_nz = torch_npu.npu_format_cast(
    wa_fp4.view(torch.uint8), torch_npu.Format.FRACTAL_NZ,
    customize_dtype=torch.float8_e4m3fn, input_dtype=torch.float4_e2m1fn_x2,
)
out = attn_epilogue(x, wa_nz, wb_nz, sa_fp4, sb_kn.view(torch.float8_e8m0fnu), sin, cos)
```

FP4 NZ 的调用入口通过同址 `[NG,128,16384]` 字节视图搬运 `C0_16` 权重。

## 图模式

支持 `torch.npu.NPUGraph`，也支持 `torch.compile` 的 `npugraph_ex` 后端。
模型中可将量化权重及 scale 注册为固定的 buffer，保留权重的 NZ 存储。

```python
class Projection(torch.nn.Module):
    """Keep quantized weights as fixed graph parameters."""

    def __init__(self, woa, wob, sa, sb):
        """Retain the initialized NPU tensors without converting their formats."""
        super().__init__()
        for name, value in zip(("woa", "wob", "sa", "sb"), (woa, wob, sa, sb)):
            self.register_buffer(name, value)

    def forward(self, x, sin, cos):
        """Project the current activations with the fixed quantized weights."""
        return attn_epilogue(x, self.woa, self.wob, self.sa, self.sb, sin, cos)


model = torch.compile(
    Projection(woa, wob, descale_woa, descale_wob).eval(),
    backend="npugraph_ex", fullgraph=True, dynamic=False,
)
out = model(x, rope_sin, rope_cos)
```

同一模型跨 T 使用时可设 `dynamic=True`。
图回放的输出缓冲遵循后端的静态复用语义；跨回放保留结果时，先用 `out.clone()` 保存。

若直接编译七参数函数，并把 NZ 权重作为调用实参，使用后端的 `clone_input=False`。

```python
compiled_op = torch.compile(
    attn_epilogue, backend="npugraph_ex", fullgraph=True, dynamic=False,
    options={"clone_input": False},
)
out = compiled_op(x, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos)
```

此时使用固定输入缓冲区更新数据，保持 NZ 权重的存储地址不变；替换权重存储后重新建图。
当前 torch_npu 后端默认克隆用户输入时可能把 NZ 转为 ND，随后尝试不支持的 FP8
NZ 格式复制。该过程位于算子外部。
ND 实参可使用后端默认配置。算子不修改全局格式开关，也不在 forward 中重新打包权重。

## 测试

[test_attn_epilogue.py](../../../test/attention/attn_epilogue/test_attn_epilogue.py) 覆盖三种 NG、ND/NZ、尾块、
NPUGraph、npugraph_ex、官方转换的 FP4 WOA `FRACTAL_NZ_C0_16` 以及 Meta 输出推导。

```bash
# 使用仓库公共 marker 选择 CPU / NPU 阶段。
python3 -m pytest -q test/attention/attn_epilogue -m 'not npu'
ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
  python3 -m pytest -q test/attention/attn_epilogue -m npu
```

当前代码要求 `cannbot-dsl>=0.8.2`，以及匹配的 CANN 和 torch_npu。
完整精度、压力、性能和原始 profiling 由仓外开发验证脚本维护，线上 CI 不收集这些脚本。

## 性能对比（DSL 0.8.0）

2026-11-02 在 Ascend950DT 上完成 16 个代表点复测，使用包含片上容量约束修正的实现、
CANN 9.3.0、DSL 0.8.0、torch 2.12.0+cpu、torch_npu 2.12.0.post2，设备为
32 AIC / 64 AIV、128 MiB L2。本次未重新执行下面历史记录中的 576 点全量扫描。

表中耗时为设备 kernel 时长中位数，单位为微秒；原生链路取五个 kernel 的时长之和。

| 场景 | 权重布局 | T | 原生链路（µs） | 融合算子（µs） | 加速比 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| NG=8 | ND | 1 | 48.178 | 29.924 | 1.610× |
| NG=8 | ND | 72 | 56.465 | 41.449 | 1.362× |
| NG=8 | ND | 128 | 66.463 | 49.918 | 1.331× |
| NG=8 | ND | 256 | 87.355 | 73.464 | 1.189× |
| NG=8 | NZ | 1 | 45.078 | 28.239 | 1.596× |
| NG=8 | NZ | 72 | 55.338 | 40.445 | 1.368× |
| NG=8 | NZ | 128 | 63.409 | 49.397 | 1.284× |
| NG=8 | NZ | 256 | 96.873 | 73.363 | 1.320× |
| NG=2 | ND | 1 | 35.511 | 19.578 | 1.814× |
| NG=2 | ND | 16 | 37.096 | 21.400 | 1.733× |
| NG=2 | NZ | 1 | 29.942 | 18.431 | 1.625× |
| NG=2 | NZ | 16 | 31.898 | 21.635 | 1.474× |
| NG=1 | ND | 1 | 30.793 | 14.036 | 2.194× |
| NG=1 | ND | 16 | 34.368 | 17.791 | 1.932× |
| NG=1 | NZ | 1 | 26.480 | 13.266 | 1.996× |
| NG=1 | NZ | 16 | 30.030 | 16.925 | 1.774× |

原生五 kernel 链路与融合单 kernel 使用相同输入、相同 ND/NZ 权重，交错进行图回放；
每点每侧 48 个样本，调用前在同流读取 256 MiB 缓冲区进行软件 L2 驱逐。
计时排除驱逐、编译、输入恢复和主机调度；加速比为原生中位数 / 融合中位数。
该口径不表示端到端请求时延。16 点全部获得收益，加速比为 1.189–2.194×，
普通执行及每点 16 个图捕获输出均与原生参考逐位一致。

## 性能对比（DSL 0.7.1 历史全量结果）

以下为 2026-10-26 在 Ascend950DT_9582 上的实测结果，测量代码对应提交
`d97f4acd2114237d900d2087565efad68fed3b62`。设备为 32 AIC / 64 AIV、128 MiB L2，
软件环境为 CANN 9.3.0（V100R001C12B140）、DSL 0.7.1、torch 2.12.0+cpu、
torch_npu 2.12.0.post2。

本节保留此前 DSL 0.7.1 的 576 点测量记录，不将其作为 DSL 0.8.0 的新测结果。

原生链路依次执行 inverse RoPE、第一次 MX 量化、第一级 Matmul、第二次 MX 量化、
第二级 Matmul。原生与融合使用相同输入、量化权重及 ND/NZ 布局，交错进行图重放；
每点每侧采集 48 个样本。每次调用前同流读取 256 MiB 缓冲区进行 ReduceSum 软件 L2 驱逐，
软件驱逐不等同于硬件 cache invalidate。

原生耗时取五个 kernel 的设备时长之和，融合耗时取单个 kernel 的设备时长，均统计中位数；
加速比为原生中位数 / 融合中位数。计时不包含输入恢复、L2 驱逐、权重预处理、编译、
内存分配、Python 调度及核间调度间隙，不表示端到端请求时延。

| 场景 | 权重布局 | T 范围 | 点数 | 加速比范围 | 加速比几何平均 |
| :--- | :--- | :--- | ---: | ---: | ---: |
| NG=8 | ND | 1–256 | 256 | 1.149–1.577× | 1.284× |
| NG=8 | NZ | 1–256 | 256 | 1.213–1.440× | 1.297× |
| NG=2 | ND | 1–16 | 16 | 1.618–1.860× | 1.697× |
| NG=2 | NZ | 1–16 | 16 | 1.412–1.620× | 1.491× |
| NG=1 | ND | 1–16 | 16 | 1.772–2.208× | 1.891× |
| NG=1 | NZ | 1–16 | 16 | 1.472–1.763× | 1.591× |

576 个点全部快于对应原生链路；每点均核对普通执行及 16 个图捕获输出，BF16 位模式全部一致。

## Native 安装包

共享 [Native 构建工程](../../../packaging/README.md) 的源码清单包含接口、切分和 kernel 三个文件。

按打包说明准备环境后，在仓库根目录构建并安装。下例使用 Python 3.12、Linux x86_64，具体文件名以构建输出的 `Wheel:` 路径为准。

```bash
python3 packaging/native/build.py --operator=attn_epilogue
python3 -m pip install --force-reinstall --no-deps packaging/native/output/cannbot_dsl_ops-0.3.0-cp312-cp312-linux_x86_64.whl
```

安装后通过 `from ops.attn_epilogue import attn_epilogue` 导入。默认导出 48 个配置，覆盖 28/32 核、
双 FP8 的 ND/ND 和 NZ/NZ，以及 FP4 WOA 搭配 FP8 WOB 的 NZ/NZ。
NG=8 的默认 T 为 `72,80,96,120`，NG=1/2 的默认 T 为 `1,16`。
其他规格默认回退 JIT。可用 `CANNBOTDSL_DS41_PROFILES` 指向一个配置文件，覆盖默认集合。

```json
{"attn_epilogue":[{"block_num":32,"ng":2,"first_fp4":true,"weight_format":"nz","tokens":[1,7,16]}]}
```

`weight_format` 同时指定两级权重的布局。FP4 WOA 只接受 `nz`。
可用 `CANNBOTDSL_NATIVE_BINARY_MODE=require` 检查二进制覆盖，用
`CANNBOTDSL_NATIVE_BINARY_REPORT=/path/to/report.json` 保存命中记录。
本次重构调整了内部 ABI，已有 Native 资源应使用当前源码和匹配运行环境的 DSL 重新构建。
