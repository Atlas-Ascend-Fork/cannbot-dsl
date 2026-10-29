# flash_kda_cp_main

依赖版本：CANNBot-DSL 0.8.0。

## 产品支持情况

- Ascend 950：支持。

## 功能说明

`flash_kda_cp_main` 基于 CANNBot-DSL 实现 Kimi Delta Attention 的 CP prefill 后半段，读取 `flash_kda_cp_preprocess` 生成的块内张量和外部 CP prefix scan 生成的本 rank 初始状态，输出本 rank 序列结果与最终状态。实现位于 [`flash_kda_cp_main.py`](flash_kda_cp_main.py)。

调用时先生成调度 metadata，再执行计算：

1. [`flash_kda_cp_main_metadata`](flash_kda_cp_main_metadata.py)：根据输入形状、布局和有效核数生成分轮调度信息。
2. `flash_kda_cp_main`：消费 metadata，启动 AICore 算子，完成跨块状态递推与输出。

### 计算过程

将序列按 64 个 token 分块。对第 $c$ 块，令 $S_{c-1}$ 为进入该块的状态，`qg`、`kg`、`u`、`w`、`mqk`、`decay` 为 `flash_kda_cp_preprocess` 的输出（`qg` 已含 QK scale，`w` 已含负号，`mqk` 已含对角下三角 mask）：

$$
\Delta_c=u_c+w_cS_{c-1}^{\top},\qquad
O_c=q^g_cS_{c-1}^{\top}+M_{qk,c}\Delta_c,
$$

$$
S_c=S_{c-1}\operatorname{diag}(\mathrm{decay}_c)+\Delta_c^{\top}k^g_c.
$$

这里状态的存储形状为 `[D, D]`，`D = 128`。主算子直接消费预处理施加过 mask 的 `mqk`，不再生成 mask；最后一个不足 64 行的 chunk 只写有效行。`local_initial_state` 由外部 CP prefix scan 用预处理的 `(transition, state_ext)` 与全局初始状态合成，本接口从它开始递推。

## 函数原型

先调用同目录 metadata 脚本的 `flash_kda_cp_main_metadata(qg, local_initial_state, layout_qkv, cu_seqlens)`，再将结果作为 `metadata` 传入主接口。

```python
flash_kda_cp_main.flash_kda_cp_main.flash_kda_cp_main(
    transition: torch.Tensor, state_ext: torch.Tensor, decay: torch.Tensor,
    qg: torch.Tensor, kg: torch.Tensor, u: torch.Tensor, w: torch.Tensor,
    mqk: torch.Tensor, local_initial_state: torch.Tensor, layout_qkv: str,
    metadata: torch.Tensor, cu_seqlens: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]
```

模块路径为 `flash_kda_cp_main.flash_kda_cp_main`。除 `cu_seqlens` 外，参数均为必选，返回值为 `(out, final_state)`。

## 参数说明

`B` 为逻辑序列数，`Nv` 为 Value 头数，`S` 为 BNSD/BSND 每条序列的 token 数，`T` 为 TND 打包后的总 token 数，`D = 128` 为 Query/Key 和 Value 的共同头维度。统一 chunk 容量 `C` 在 BNSD/BSND 下为 `ceil(S / 64) + B - 1`，在 TND 下为 `ceil(T / 64) + B - 1`。`M` 为 metadata factory 返回的 word 容量。BNSD/BSND 的中间 token 张量均使用 BNSD 布局，TND 中间张量保持打包布局；metadata 按每条逻辑序列实际的 `ceil(length_i / 64)` 调度。主接口内部按需补齐到 64 行，返回的 token 轴长度仍为 `S` 或 `T`。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|:---|:---|:---|:---|:---|:---|
| `transition` | Tensor | 必选 | rank 局部状态仿射变换，供外部 CP prefix scan 求 `local_initial_state`；本接口校验其 shape 和 dtype。 | float32 | `[B, Nv, D, D]` |
| `state_ext` | Tensor | 必选 | rank 局部状态仿射加项，用途同上。 | float32 | `[B, Nv, D, D]` |
| `decay` | Tensor | 必选 | 每块末端逐通道状态衰减；metadata 中每条序列只消费自己的有效 chunk。 | float32 | `[B, Nv, C, D]` |
| `qg` | Tensor | 必选 | 预处理后的衰减 Query，已包含 QK scale；BNSD/BSND 输入均传 BNSD。 | bfloat16 | BNSD/BSND：`[B, Nv, S, D]`；TND：`[T, Nv, D]` |
| `kg` | Tensor | 必选 | 预处理后的状态更新 Key。 | bfloat16 | BNSD/BSND：`[B, Nv, S, D]`；TND：`[T, Nv, D]` |
| `u` | Tensor | 必选 | 预处理后修正的 Value。 | bfloat16 | BNSD/BSND：`[B, Nv, S, D]`；TND：`[T, Nv, D]` |
| `w` | Tensor | 必选 | 预处理后带负号的 Key 修正权重；在 $\Delta_c=u_c+w_cS_{c-1}^{\top}$ 中相加。 | bfloat16 | BNSD/BSND：`[B, Nv, S, D]`；TND：`[T, Nv, D]` |
| `mqk` | Tensor | 必选 | 块内含对角下三角 QK 矩阵，末块无效位置不参与计算，尾部容量槽保持未消费。 | bfloat16 | `[B, Nv, C, 64, 64]` |
| `local_initial_state` | Tensor | 必选 | 外部 CP prefix scan 产生的本 rank 初始状态。 | float32 | `[B, Nv, D, D]` |
| `layout_qkv` | str | 必选 | 输出布局，取值为 `"BNSD"`、`"BSND"` 或 `"TND"`；BSND 仅影响输出，稠密中间张量仍使用 BNSD。 | - | - |
| `metadata` | Tensor | 必选 | 同目录 metadata 算子返回的调度张量。 | int32 | `[M]`，容量 `M` 由 metadata 算子确定 |
| `cu_seqlens` | Tensor | TND 必选，其他布局为 None | TND 逻辑序列边界，首项为 0、末项为 `T`，严格递增，每条序列长度为正。 | int32 | `[B + 1]` |

`transition` 和 `state_ext` 是 CP 前缀扫描的输入，本主算子接收它们以保持接口一致并验证输入；实际递推从 `local_initial_state` 开始。

### metadata 入参

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|:---|:---|:---|:---|:---|:---|
| `qg` | Tensor | 必选 | 与主算子相同的衰减 Query；检查 shape、dtype、设备和连续性，不读取数值。 | bfloat16 | BNSD/BSND：`[B, Nv, S, D]`；TND：`[T, Nv, D]` |
| `local_initial_state` | Tensor | 必选 | 与主算子相同的初始状态；检查 shape、dtype、设备和连续性，不读取数值。 | float32 | `[B, Nv, D, D]` |
| `layout_qkv` | str | 必选 | 与主算子一致的输出布局。 | - | - |
| `cu_seqlens` | Tensor | TND 必选，其他布局为 None | 与主算子一致的 TND 序列边界，严格递增，从 0 到 `T`。 | int32 | `[B + 1]` |

metadata 算子读取 `cu_seqlens` 边界值；对于 `qg` 和 `local_initial_state`，仅检查张量属性，不读取数值。

### metadata 复用

相同 `B`、`Nv`、`S`（稠密）或 `T` 与序列边界（TND）、布局、当前 stream 有效核数及内置调度 cost 公式下，可跨层复用 metadata；上述条件或 cost 公式版本变化后须重新生成。

metadata 记录每条逻辑序列实际消费的 chunk 数；`decay` 和 `mqk` 的 chunk 轴按同一模块导出的 `chunk_capacity_upper_bound(B, S)`（稠密）或 `chunk_capacity_upper_bound(B, T)`（TND）申请，容量为上文定义的 `C`。

主算子的 metadata wrapper 与 CP preprocess 使用同一个共享 CP ABI 和 AICPU 编译对象。主算子跳过共享 ABI 中为预处理调度保留的字段。

## 返回值说明

返回 `(out, final_state)`，两个张量均位于输入所在 NPU。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|:---|:---|:---|:---|:---|:---|
| `out` | Tensor | 必选 | 本 rank 序列输出。 | bfloat16 | BNSD：`[B, Nv, S, D]`；BSND：`[B, S, Nv, D]`；TND：`[T, Nv, D]` |
| `final_state` | Tensor | 必选 | 本 rank 递推结束后的状态，不原地修改输入。 | float32 | `[B, Nv, D, D]` |

BSND 最终 `out` 的逻辑形状为 `[B, S, Nv, D]`，通过内部 BNSD 存储的转置视图返回，不保证连续；CP 中间 token 量统一使用 BNSD。TND 输入对应 TND 输出。

## 约束说明

- 支持 BNSD、BSND 和多序列 TND 输出布局以及 `D = 128`；`B >= 1`、`Nv >= 1`、稠密 `S > 0`、TND `T > 0`，每个序列末块可不足 64 行。`decay/mqk` 的 chunk 轴使用上文定义的统一容量 `C`，metadata 负责限制每个逻辑序列的有效范围。
- TND 的 `cu_seqlens` 形状为 `[B + 1]`，首项为 0、末项为 `T`，严格递增，每条逻辑序列长度为正；metadata 和主算子在一次下发中按逻辑序列复用同一套共享 CP 调度。
- 所有输入张量（包括 `metadata`）必须连续、位于同一 NPU，且满足参数表中的 dtype 和 shape。
- metadata 和主算子必须使用相同的当前 stream 有效核数；内核的全栅栏同步依赖此约定。
- 外部 CP prefix scan 负责把预处理的 `(transition, state_ext)` 与全局初始状态合成为 `local_initial_state`。
- 完整 chunk 使用双行 BF16 向量快路径；不足 64 行的尾块执行有效行清零和掩码拷贝。

## 调用示例

```python
import torch
import torch_npu
from flash_kda_cp_main.flash_kda_cp_main import flash_kda_cp_main
from flash_kda_cp_main.flash_kda_cp_main_metadata import flash_kda_cp_main_metadata

B, Nv, S, D = 1, 1, 64, 128
qg = torch.zeros(B, Nv, S, D, dtype=torch.bfloat16).npu()
kg = torch.zeros_like(qg)
u = torch.zeros_like(qg)
w = torch.zeros_like(qg)
mqk = torch.zeros(B, Nv, 1, 64, 64, dtype=torch.bfloat16).npu()
decay = torch.ones(B, Nv, 1, D, dtype=torch.float32).npu()
local_initial_state = torch.zeros(B, Nv, D, D, dtype=torch.float32).npu()
transition = torch.eye(D, dtype=torch.float32).reshape(B, Nv, D, D).npu()
state_ext = torch.zeros_like(local_initial_state)

metadata = flash_kda_cp_main_metadata(qg, local_initial_state, "BNSD")
out, final_state = flash_kda_cp_main(transition, state_ext, decay, qg, kg, u, w,
    mqk, local_initial_state, "BNSD", metadata)
```

## 精度测试

在仓库根目录执行单文件测试。测试文件内置独立 FP32 块递推 golden，
以随机输入比对输出和最终状态，保留 BNSD、BSND、TND 三个白盒精度场景和一个最小全量场景，
容差为 `atol=3e-3, rtol=3e-2`；另覆盖最小 metadata 调度容量：

```bash
python -m pytest -q test/flash_kda_cp_main/test_flash_kda_cp_main.py
```

## 性能

测试在 Ascend 950、CANN 9.2.0 上用 `msprof --task-time=l1` 采集，形状为 `B=1, Nqk=32, Nv=32, S=8192, D=128`，覆盖 BNSD、BSND、TND 三种布局。

metadata 在采样循环前生成一次，`Task Duration` 只统计 AICore kernel `flash_kda_cp_kernel`。每种布局预热 2 次后连续采样 12 次，表中为 12 次采样的中位数。

| Layout | Task Duration 中位数 (us) |
|:---|--:|
| BNSD | 280.5 |
| BSND | 280.2 |
| TND | 282.5 |
