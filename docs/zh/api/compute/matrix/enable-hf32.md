# enable_hf32
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

本接口用于设置Mmad计算开启HF32模式，开启该模式后，Mmad计算FP32数据的性能将得到提升，但会带来一定的精度损失。

是否开启HF32对Mmad计算理论性能的影响见下表：

**表1** HF32对Mmad计算理论性能的影响

| 接口 | 左矩阵A | 右矩阵B | $cube_m$ | $cube_n$ | $cube_k$ | $k_0$ |
| --- | --- | --- | --- | --- | --- | --- |
| `matmul`（不开启HF32） | `dtypes.float32` | `dtypes.float32` | 16 | 16 | 1 | 8 |
| `matmul`（开启HF32） | `dtypes.float32` | `dtypes.float32` | 16 | 16 | 8 | 8 |

性能计算公式如下：

$$
\begin{gathered}
{ceil_m} = \left\lceil \frac{m}{16} \right\rceil \times 16 \\[12pt]
{ceil_n} = \left\lceil \frac{n}{16} \right\rceil \times 16 \\[12pt]
{ceil_k} = \left\lceil \frac{k}{k_0} \right\rceil \times k_0 \\[16pt]
\text{cube利用率} =
\frac{ (m \times n \times k) / ({cube_m} \times {cube_n} \times {cube_k}) }
{ \Delta t + ({ceil_m} \times {ceil_n} \times {ceil_k}) / ({cube_m} \times {cube_n} \times {cube_k}) }
\end{gathered}
$$

关键变量及常量说明：

- $m, n, k$：mmad入参实际计算的大小。
- $ceil_m, ceil_n, ceil_k$：$m, n, k$根据分形大小向上对齐后的值。
- $cube_m, cube_n, cube_k$：硬件真实并行度（单位：elements/cycle）。
- $k_0$：L0 Buffer上最小分形K方向大小。
- $\Delta t$：头开销cycle数。

开启HF32模式后，L0A Buffer/L0B Buffer中的FP32数据将在参与Mmad计算之前被舍入为HF32格式，舍入模式使用`set_hf32_round_mode`接口配置，中间计算使用HF32格式，最终的运算结果仍以FP32格式输出，以保证后续处理的兼容性。

FP32与HF32格式的精度对比如下图所示：

**图1** FP32与HF32格式精度示意图

![FP32与HF32格式精度示意图](../../figures/mmad_hf32_950.svg)

## 函数原型

```python
enable_hf32() -> None
```

## 参数说明

无。

## 返回值说明

无返回值。

## 约束说明

- 本接口运行在`[PIPE](../../execution/pipeline.md).S`流水上。
- 本接口需在矩阵乘加指令（`matmul`）执行前调用，以此来确保模式配置在矩阵乘加计算过程中生效。
- HF32模式启用后会持续生效，不会自动关闭。后续矩阵乘加指令若不显式重新配置，将沿用当前模式。如需关闭HF32模式，请重新调用`set_fp32_mode`接口。
- HF32模式启用后，FP32转换为HF32格式的舍入模式由配套的`set_hf32_round_mode`接口配置，须在本接口之后调用该配套接口配置舍入模式，否则舍入模式沿用上次配置或默认值。
- HF32模式仅对输入矩阵数据类型为`dtypes.float32`的矩阵乘加运算场景生效。

## 调用示例

以下示例演示`enable_hf32`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, dtypes, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.cube import enable_hf32
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.tensor import MemLoc

# HF32 的尾数为 10 bit，1.0 与 1 + 2^-10 的中点即 1 + 2^-11，恰好落在舍入的平局点上。
TIE = 1.0 + 0.5 / 1024.0

@kernel
def _enable_hf32_kernel(a, b, c):
    l1a = Buffer(MemLoc.L1, dtypes.float32, make_layout((16, 16)))
    l1b = Buffer(MemLoc.L1, dtypes.float32, make_layout((16, 16)))
    l0a = Buffer(MemLoc.L0A, dtypes.float32, make_layout((16, 16)))
    l0b = Buffer(MemLoc.L0B, dtypes.float32, make_layout((16, 16)))
    l0c = Buffer(MemLoc.L0C, dtypes.float32, make_layout((16, 16)))
    mem_copy(l1a, a)
    mem_copy(l1b, b)
    enable_hf32()
    mem_copy(l0a, l1a)
    mem_copy(l0b, l1b)
    result = matmul(l0c, l0a, l0b, init=True)
    mem_copy(c, result)

@host
def run(a, b, c):
    _enable_hf32_kernel[1](a, b, c)

def main():
    a = torch.eye(16, dtype=torch.float32, device="npu:0") * TIE
    b = torch.eye(16, dtype=torch.float32, device="npu:0")
    c = torch.empty((16, 16), dtype=torch.float32, device="npu:0")
    run(a, b, c)
    torch.npu.synchronize()

    # 开启HF32模式后，L0A Buffer/L0B Buffer中的float数据先舍入为HF32格式再参与Mmad计算：
    # 未调用set_hf32_round_mode时默认使用NEAREST_EVEN，TIE被舍入为1.0；
    # 未开启HF32模式时不做舍入处理，结果即为TIE本身。
    torch.testing.assert_close(c.cpu(), torch.eye(16, dtype=torch.float32))
    print(f"tie={a.cpu()[0, 0].item():.8f} -> hf32={c.cpu()[0, 0].item():.8f}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
tie=1.00048828 -> hf32=1.00000000
```
