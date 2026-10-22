# IndexerPrologueQw

融合 MXFP8 Q GEMM → 尾部 RoPE → MXFP4（E2M1 + E8M0）量化，以及 BF16 W GEMM × `softmax_scale`。固定几何为 `dim=5120, q_lora=1280, N=32, D=128, Dr=64`，`T=batch*seq` 为动态轴。

建议阅读顺序：公开入口 `indexer_prologue_qw()` → `Plan` → 主 kernel → `CubeQ` / `CubeW` / `VectorQ`。关键函数上方有中文职责、参数和数据流说明。

## 文件与接口

**`indexer_prologue_qw.py`** 与 `cannbot-arena-ds41/net/ops/indexer_prologue_qw.py` 执行代码一致，另补充中文注释和说明文字。类名、算子注册名和调参环境变量统一为 `IndexerPrologueQw`、`indexer_prologue_qw` 和 `IPQW_*`。测试直接导入 `indexer_prologue_qw.py`；`__init__.py` 直接导出源文件的 NPU-only 入口，不提供 CPU fallback。CPU 参考计算、部署矩阵和验证入口统一在 `test/indexer_prologue_qw/ipqw_verify.py`。


保留文件的用途：

| 文件 | 用途 |
| --- | --- |
| `indexer_prologue_qw.py` | 完整算子实现，与指定源文件保持相同执行逻辑，并附中文注释 |
| `__init__.py` | 包入口，仅导出算子和权重格式转换工具 |
| `../../test/indexer_prologue_qw/ipqw_verify.py` | 共用 CPU 参考计算、输入构造、部署矩阵、精度检查和性能计时 |

NPU 上 `wqb`、`ww` 必须在加载权重时调用一次 `to_nz()` 转为 FRACTAL_NZ。其余输入保持 ND，短尾在 L1 内处理，无需调用者补齐行数。`T` 从 `qr` 获取，允许 `x` 的行数大于 T，多余行忽略。

| 默认模板 | T 范围 | 调度与 W 归约 |
| --- | --- | --- |
| split-K | 1–3072 | 固定 16 个 head 分块；W 的 K 轴切分，workspace 内确定性归约 |
| split-T | 3073–262144 | 沿 T 切分；每个 T tile 独立完成全部 head |

两个模板都使用动态 T，默认 `base_m=128`。模板边界由 `IPQW_SPLIT_T_TILES=25` 决定，运行时 grid 使用设备 Cube 核数。本次验证设备为 Ascend950PR，36 Cube / 72 Vector 核。

## 运行

```bash
source /xxxx/workspaces/tools/ENTER/etc/profile.d/conda.sh
conda activate xxxx
source /xxxx/codex/AscendC/cann/bin/setenv.bash
# 在仓库根目录运行；test 路径仅用于下面示例的输入构造工具。
export PYTHONPATH="$PWD/samples:$PWD/test/indexer_prologue_qw${PYTHONPATH:+:$PYTHONPATH}"
```

```python
import torch_npu
from indexer_prologue_qw import indexer_prologue_qw, to_nz
from ipqw_verify import make_inputs

inputs = make_inputs(72, 5120, 1280, 32, 128, 64,
                     softmax_scale=128**-0.5, seed=0)
q, descale_q, w = indexer_prologue_qw(
    inputs.x.npu(), inputs.qr.npu(),
    to_nz(inputs.wqb.npu()), to_nz(inputs.ww.npu()),
    inputs.descale_qr.npu(), inputs.descale_wqb.npu(),
    inputs.rope_sin.npu(), inputs.rope_cos.npu(),
    softmax_scale=inputs.softmax_scale,
)
```

| 张量 | 形状 / 类型 |
| --- | --- |
| x | `(T, 5120)` BF16，可有额外行 |
| qr | `(T, 1280)` FP8 E4M3 或 uint8 字节视图 |
| wqb | `(4096, 1280)` FP8 E4M3 / uint8，NPU FRACTAL_NZ |
| ww | `(32, 5120)` BF16，NPU FRACTAL_NZ |
| descale_qr / descale_wqb | `(T, 20, 2)` / `(4096, 20, 2)` paired E8M0 |
| rope_sin / rope_cos | `(T, 64)` FP32，interleave RoPE；相邻 `(2i, 2i+1)` 通道构成一对，表值也按对重复 |
| q | `(T, 32, 64)` uint8，低 nibble 为偶数列，高 nibble 为奇数列 |
| descale_q | `(T, 32, 2, 2)` E8M0 字节 |
| w | `(T, 32)` FP32 |

可选 `q`、`descale_q`、`w` 参数接收预分配输出；无需预先清零。

## 验证

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python test/indexer_prologue_qw/ipqw_verify.py \
  --output /tmp/indexer_prologue_qw_validation.json

# 仅复测 T=72、2048 性能
python test/indexer_prologue_qw/ipqw_verify.py --perf-only --output /tmp/indexer_prologue_qw_perf.json

OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 pytest test/indexer_prologue_qw -q
```

测试代码统一放在 `test/indexer_prologue_qw/`，按职责保留四个 pytest 文件：

| 文件 | 覆盖内容 |
| --- | --- |
| `test_golden.py` | CPU 参考计算、输出约定和量化误差检查器 |
| `test_vf_math.py` | Vector 量化数学、舍入边界与极端数值 |
| `test_kernel_compile.py` | 调度计划、资源约束和两个模板的编译 |
| `test_deployment.py` | NPU 部署矩阵、尾块、RoPE/scale、多 seed、重复性和超长 x |

原 `test_npu_precision.py` 的全部用例已并入 `test_deployment.py`。
原 `golden.py`、`shapes.py`、`verify.py` 已合并为测试目录中的 `ipqw_verify.py`，
参考计算、部署矩阵及精度判断各维护一份，pytest 直接复用。
只需精度回归时运行 pytest；需要性能或 JSON 报告时运行 `ipqw_verify.py`，无需为相同精度矩阵再跑两遍。

精度矩阵：

- Decode：batch=`1,4,8,12,16,32`，seq=`1,6`（MTP=5）。
- Prefill：batch=`1,4,8,16,32`，seq=`1024,2048,4096,8192`。

共 32 个 batch/seq 组合，按算子实际输入 T 去重后为 21 个形状。所有形状都对全部行进行 CPU golden 对比；golden 每次计算 1024 行以限制内存，不做抽样。pytest 另覆盖非对齐尾部、重复调用逐位一致性及超长 x 的死行隔离。

Q 和 descale 默认逐字节检查；仅允许源实现已有的 FP32 累加顺序引起的量化边界差异（相邻 E2M1 档位、零符号及 E8M0 边界），并在 JSON 中记录原始差异数。W 使用 `rtol=atol=2e-2`，同时记录最大绝对误差。

性能采用 NPU graph 内连续 100 次调用、10 轮 NPU Event 计时，报告每次调用的 min/median/max。计时包含图执行中的设备调度开销，不包含编译、CPU→NPU 搬运、权重 NZ 转换及 Python 调用开销；不等同于 profiler 的单 kernel duration。
