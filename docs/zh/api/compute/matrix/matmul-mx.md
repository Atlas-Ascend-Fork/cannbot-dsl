# matmul（MX模式）

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

本接口用于完成带有量化系数的矩阵乘加计算（MmadMx），即左矩阵A和右矩阵B均有对应的量化系数矩阵，分别为左量化系数矩阵ScaleA和右量化系数矩阵ScaleB。在MX模式中，左量化系数矩阵与左矩阵相乘，右量化系数矩阵与右矩阵相乘，然后对两个乘积的结果做矩阵乘法，并加上初始累加矩阵，其计算公式如下：

$$
\mathrm{C}_{M \times N} = (\mathrm{ScaleA}_{M \times (K/32)} \otimes \mathrm{A}_{M \times K}) \times (\mathrm{ScaleB}_{(K/32) \times N} \otimes \mathrm{B}_{K \times N}) + \mathrm{C}_{M \times N}
$$

其中，$\otimes$表示广播乘法，矩阵A、B沿K维度每连续32个元素共享对应量化系数矩阵ScaleA、ScaleB中的一个量化系数。C为初始累加矩阵，可以通过配置`init`和`bias`，初始化为全0矩阵、L0C Buffer中的矩阵或广播后的Bias矩阵。本页描述`matmul`接口在`mx_mode=True`时的行为，非量化模式矩阵乘加计算请参见[matmul（普通模式）](./matmul.md)。各矩阵的信息说明见下表：

**表** 矩阵信息说明

| 矩阵 | 存储位置 | 形状（行数×列数） | 数据格式 | 分形大小（行数×列数） |
| --- | --- | --- | --- | --- |
| A | L0A Buffer | M×K | Nz | 16×K0 |
| ScaleA | L0A_MX Buffer | M×(K / 32) | Zz | 16×2 |
| B | L0B Buffer | K×N | Zn | K0×16 |
| ScaleB | L0B_MX Buffer | (K / 32)×N | Nn | 2×16 |
| C | L0C Buffer | M×N | Nz | 16×16 |
| Bias（用于C矩阵初始化） | BiasTable Buffer | 1×N，使用时通过广播复制M行来初始化C矩阵 | ND | - |

当矩阵数据类型为`dtypes.float8_e5m2`、`dtypes.float8_e4m3fn`时，K0为32；当矩阵数据类型为`dtypes.fp4x2_e2m1`、`dtypes.fp4x2_e1m2`时，K0为64。实际搬运时，量化系数矩阵和输入矩阵的物理存储还需按[内存使用约束](#内存使用约束)补齐。

本接口不单独传入Scale矩阵地址，而是根据A、B矩阵的起始地址，按照固定的地址映射关系从L0A_MX Buffer和L0B_MX Buffer读取对应的量化系数矩阵。调用前，通过[`mem_copy`](../../data-movement/mem-copy.md)将左右输入从L1搬入L0A/L0B，并分别使用[`copy_options(mx_scale=...)`](../../data-movement/copy-options.md#参数说明)提供位于L1的ScaleA和ScaleB视图，具体见[调用示例](#调用示例)。

输入矩阵的排布参见[关键分形格式详解](./fractal-formats.md)，Bias与Scale矩阵的排布参见[辅助矩阵分形格式详解](./aux-fractal-formats.md)；数据准备、计算和结果搬出的衔接参见[矩阵计算流程](./flow.md)。

## 函数原型

```python
matmul(dst: Tensor, lhs: Tensor, rhs: Tensor, *, bias: Tensor | None = None, init = True, unit_flag: int = 0, disable_gemv: bool = True, mx_mode: bool = False) -> Tensor
```

使用MX模式时，必须显式传入`mx_mode=True`；省略该参数会使用普通模式。

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `dst` | 输入/输出 | 目的操作数，结果矩阵C位于L0C Buffer，数据类型为`dtypes.float32`，起始地址需按照1024字节对齐。初始累加值由`init`和`bias`共同决定，见[初始化与累加](./matmul.md#初始化与累加)。 |
| `lhs` | 输入 | 源操作数，左矩阵A位于L0A Buffer。<br>FP8类型（`dtypes.float8_e4m3fn`、`dtypes.float8_e5m2`）的起始地址需按照1024字节对齐；FP4类型（`dtypes.fp4x2_e2m1`、`dtypes.fp4x2_e1m2`）的起始地址需按照512字节对齐。ScaleA在L0A_MX Buffer中的起始地址为A起始地址除以16。数据类型组合请参见[支持的数据类型](#支持的数据类型)。 |
| `rhs` | 输入 | 源操作数，右矩阵B位于L0B Buffer。<br>FP8类型（`dtypes.float8_e4m3fn`、`dtypes.float8_e5m2`）的起始地址需按照1024字节对齐；FP4类型（`dtypes.fp4x2_e2m1`、`dtypes.fp4x2_e1m2`）的起始地址需按照512字节对齐。ScaleB在L0B_MX Buffer中的起始地址为B起始地址除以16。数据类型组合请参见[支持的数据类型](#支持的数据类型)。 |
| `bias` | 输入 | Bias矩阵在BiasTable Buffer中的起始地址，数据类型为`dtypes.float32`，需按照64字节对齐，可以为非零地址。初始化时，将Bias沿M方向广播，作为初始累加值。调用前，需通过`mem_copy`将Bias数据从L1 Buffer搬入。Bias占用长度为`N × 4`字节向上补齐到64字节的整数倍。默认值为`None`，表示不使用Bias。 |
| `init` | 输入 | 配置初始累加值，接受Python布尔值或运行时布尔标量，默认值为`True`。<br>&nbsp;&nbsp;&bull; `True`：重新初始化；未提供`bias`时初始值为0，提供时初始值为广播后的Bias。<br>&nbsp;&nbsp;&bull; `False`：使用`dst`中原有矩阵继续累加。<br>传入Python布尔值`False`时不能同时提供`bias`；运行时布尔标量允许在初始化分支使用Bias、在累加分支不再使用Bias。完整组合见[初始化与累加](./matmul.md#初始化与累加)。 |
| `unit_flag` | 输入 | 用于控制矩阵乘加指令与矩阵搬出指令的细粒度并行，开启UnitFlag后，硬件每计算完一个分形，计算结果就会被搬出。取值说明如下：<br>&nbsp;&nbsp;&bull; `0`（DISABLE）：不开启UnitFlag。<br>&nbsp;&nbsp;&bull; `2`（ENABLE_KEEP）：开启UnitFlag，硬件执行完指令后不改变单元标志位。<br>&nbsp;&nbsp;&bull; `3`（ENABLE_UPDATE）：开启UnitFlag，硬件执行完指令后改变单元标志位。<br>矩阵乘加指令与对应的矩阵搬出指令必须都开启或都不开启UnitFlag，开启后指令之间无需再插入同步指令。<br>同步机制和迭代场景见[UnitFlag](./key-features-unit-flag.md)。默认值为`0`。 |
| `disable_gemv` | 输入 | M为1时，配置是否关闭GEMV模式。<br>&nbsp;&nbsp;&bull; false：开启GEMV模式。<br>&nbsp;&nbsp;&bull; true：关闭GEMV模式。<br>M不为1时，该参数不生效。<br>向量布局与使用场景见[GEMV](./key-features-gemv.md)。默认值为`True`。MX输入的地址仍需满足本表中的对齐要求。 |
| `mx_mode` | 输入 | 编译期Python布尔值，接口默认值为`False`。使用本页的MX模式时必须显式设置为`True`，并在计算前为左右输入分别准备配对的量化系数矩阵。仅设置本参数不会自动生成或加载量化系数。 |

### 支持的数据类型

左右输入、量化系数矩阵、累加结果和可选Bias的数据类型组合见下表。

**表** MX模式支持的数据类型组合

| lhs | rhs | ScaleA/ScaleB数据类型 | dst | bias（可选） |
| --- | --- | --- | --- | --- |
| `dtypes.fp4x2_e1m2` | `dtypes.fp4x2_e1m2` | `dtypes.float8_e8m0` | `dtypes.float32` | `dtypes.float32` |
| `dtypes.fp4x2_e1m2` | `dtypes.fp4x2_e2m1` | `dtypes.float8_e8m0` | `dtypes.float32` | `dtypes.float32` |
| `dtypes.fp4x2_e2m1` | `dtypes.fp4x2_e1m2` | `dtypes.float8_e8m0` | `dtypes.float32` | `dtypes.float32` |
| `dtypes.fp4x2_e2m1` | `dtypes.fp4x2_e2m1` | `dtypes.float8_e8m0` | `dtypes.float32` | `dtypes.float32` |
| `dtypes.float8_e4m3fn` | `dtypes.float8_e4m3fn` | `dtypes.float8_e8m0` | `dtypes.float32` | `dtypes.float32` |
| `dtypes.float8_e4m3fn` | `dtypes.float8_e5m2` | `dtypes.float8_e8m0` | `dtypes.float32` | `dtypes.float32` |
| `dtypes.float8_e5m2` | `dtypes.float8_e4m3fn` | `dtypes.float8_e8m0` | `dtypes.float32` | `dtypes.float32` |
| `dtypes.float8_e5m2` | `dtypes.float8_e5m2` | `dtypes.float8_e8m0` | `dtypes.float32` | `dtypes.float32` |

表中`bias`列为提供Bias时的类型，不提供时传入`None`。量化系数矩阵须与对应输入的K轴分组及布局匹配。

## 返回值说明

返回本次矩阵乘加结果的`Tensor`视图，数据类型为`dtypes.float32`，与`dst`共享L0C Buffer中的存储，并包含本次计算的有效数据范围和布局信息。后续累加或搬运应使用该返回视图。

## 约束说明

### 通用约束

- 本接口仅在AIC上生效。
- `M`、`K`、`N`中的任意一个值为0时，接口将被视为NOP（空操作）。
- 必须显式传入`mx_mode=True`，且左右输入的数据类型组合须在本页的支持列表中。

### 内存使用约束

- L0C Buffer大小为256KB，L0A Buffer和L0B Buffer大小均为64KB，L0A_MX Buffer和L0B_MX Buffer大小均为4KB，BiasTable Buffer大小为4KB。矩阵的起始地址和占用空间不能超出对应Buffer的范围。
- 矩阵存储空间的M、N维需分别向上补齐到16的倍数。MX计算使用的物理K范围需向上补齐到64的倍数，补齐区域的A、B数据需为0；Tensor视图仍描述本次计算的有效范围。
- 量化系数矩阵需覆盖本次MX计算的全部K轴分组，并与输入的物理存储范围匹配。K轴分组按每64个输入元素对应2个量化系数补齐，不足64个元素按一组计。源数据及量化系数矩阵的布局和容量要求见[`copy_options`的`mx_scale`参数](../../data-movement/copy-options.md#参数说明)。
- M、K、N的单位为元素，取值范围均为[0, 4095]。
- 当M为1且`disable_gemv=False`时，L0A中的A需按ND格式排布，起始地址仍需满足本页参数表中FP8或FP4的对齐要求。

### UnitFlag约束

- 矩阵乘加指令与对应矩阵搬出指令需同时开启或同时关闭UnitFlag。多次计算后搬出结果时，前面的计算使用`unit_flag=2`（ENABLE_KEEP），最后一次计算使用`unit_flag=3`（ENABLE_UPDATE）；具体配合方式见[UnitFlag](./key-features-unit-flag.md)。
- 开启UnitFlag时，矩阵计算方向需与矩阵搬出读取顺序保持一致。矩阵搬出指令开启Nz2ND随路格式转换，或未进行随路格式转换但开启B8/B4量化并触发Channel Merge功能时，调用`cube.set_mmad_direction("n")`；其他场景调用`cube.set_mmad_direction("m")`。
- 开启UnitFlag时，建议矩阵乘加的计算数据量与矩阵搬出的数据量保持一致。两者不一致可能导致执行异常。

### 特殊值/边界值约束

- 应避免nan输入，否则可能会产生执行报错。

## 调用示例

以下示例演示FP8输入的MX矩阵乘加，并在 Host 侧校验结果。


```python
import torch
import torch_npu

from cannbotdsl import Buffer, copy_options, dtypes, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops import cube
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.tensor import MemLoc

M, K, N = 16, 64, 16
GROUPS = K // 32


@kernel
def _matmul_mx_kernel(a, b, scale_a, scale_b, bias, c):
    a_src = a.view(dtype=dtypes.float8_e4m3fn)
    b_src = b.view(dtype=dtypes.float8_e5m2)
    scale_a_src = scale_a.view(dtype=dtypes.float8_e8m0)
    scale_b_src = scale_b.view(dtype=dtypes.float8_e8m0).permute(1, 0)

    a_l1 = Buffer(MemLoc.L1, dtypes.float8_e4m3fn, make_layout((M, K)))
    b_l1 = Buffer(MemLoc.L1, dtypes.float8_e5m2, make_layout((N, K)))
    scale_a_l1 = Buffer(
        MemLoc.L1, dtypes.float8_e8m0, make_layout((GROUPS, M))
    ).permute(1, 0)
    scale_b_l1 = Buffer(
        MemLoc.L1, dtypes.float8_e8m0, make_layout((GROUPS, N))
    )
    a_l0 = Buffer(MemLoc.L0A, dtypes.float8_e4m3fn, make_layout((M, K)))
    b_l0 = Buffer(MemLoc.L0B, dtypes.float8_e5m2, make_layout((N, K)))
    c_l0 = Buffer(MemLoc.L0C, dtypes.float32, make_layout((M, N)))
    bias_l1 = Buffer(MemLoc.L1, dtypes.float32, make_layout((N,)), data_format="nd")
    bias_bt = Buffer(MemLoc.BIAS, dtypes.float32, make_layout((N,)))

    mem_copy(a_l1, a_src)
    mem_copy(b_l1, b_src)
    mem_copy(scale_a_l1, scale_a_src)
    mem_copy(scale_b_l1, scale_b_src)
    mem_copy(bias_l1, bias)
    mem_copy(bias_bt, bias_l1)

    mem_copy(a_l0, a_l1, options=copy_options(mx_scale=scale_a_l1))
    mem_copy(b_l0, b_l1, options=copy_options(mx_scale=scale_b_l1))
    cube.set_mmad_direction("n")
    result = matmul(
        c_l0, a_l0, b_l0,
        bias=bias_bt,
        init=True,
        unit_flag=3,
        disable_gemv=True,
        mx_mode=True,
    )
    mem_copy(c, result, options=copy_options(unit_flag=3))


@host
def run(a, b, scale_a, scale_b, bias, c):
    _matmul_mx_kernel[1](a, b, scale_a, scale_b, bias, c)


def main():
    # E4M3的0x38和E5M2的0x3c均表示1。
    a_bytes = torch.full((M, K), 0x38, dtype=torch.int8)
    b_bytes = torch.full((N, K), 0x3C, dtype=torch.int8)
    group = torch.arange(GROUPS, dtype=torch.int16).reshape(1, GROUPS)
    row = torch.arange(M, dtype=torch.int16).reshape(M, 1)
    col = torch.arange(N, dtype=torch.int16).reshape(N, 1)

    # E8M0量化系数为2 ** (原始字节值 - 127)。
    scale_a_exp = 127 + (row + group) % 2
    scale_b_exp = 126 + (2 * col + group) % 3
    scale_a_bytes = scale_a_exp.to(torch.uint8).view(torch.int8)
    scale_b_bytes = scale_b_exp.to(torch.uint8).view(torch.int8)
    bias = torch.arange(N, dtype=torch.float32) + 1.0

    c = torch.empty((M, N), dtype=torch.float32, device="npu:0")
    run(
        a_bytes.to("npu:0"),
        b_bytes.to("npu:0"),
        scale_a_bytes.to("npu:0"),
        scale_b_bytes.to("npu:0"),
        bias.to("npu:0"),
        c,
    )
    torch.npu.synchronize()

    a_ref = a_bytes.view(torch.float8_e4m3fn).float()
    b_ref = b_bytes.view(torch.float8_e5m2).float()
    scale_a_ref = torch.pow(2.0, scale_a_exp.float() - 127)
    scale_b_ref = torch.pow(2.0, scale_b_exp.float() - 127)
    expected = (
        a_ref * scale_a_ref.repeat_interleave(32, dim=1)
    ) @ (
        b_ref * scale_b_ref.repeat_interleave(32, dim=1)
    ).T + bias.reshape(1, N)

    actual = c.cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print(f"first={actual[0, 0].item():.3f}, last={actual[-1, -1].item():.3f}")


if __name__ == "__main__":
    main()
```

预期结果：

```text
first=81.000, last=80.000
```
