# matmul
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

本接口是面向昇腾AI芯片的矩阵乘加（Mmad）核心计算接口，专为高性能算子开发设计，封装了昇腾NPU硬件的矩阵乘加计算能力，广泛用于神经网络层（如全连接层、卷积层）、数值计算类算子的开发，其计算公式如下：

$$
\mathrm{C}_{M \times N} = \mathrm{A}_{M \times K} \times \mathrm{B}_{K \times N} + \mathrm{C}_{M \times N}
$$

其中，A、B、C分别为左、右、结果矩阵，C为初始累加矩阵，可以通过配置`init`和`bias`，初始化为全0矩阵、L0C Buffer中的矩阵或广播后的Bias矩阵，具体见[初始化与累加](#初始化与累加)。本页描述`matmul`接口在`mx_mode=False`时的行为。带有量化系数的矩阵乘加计算请参见[matmul（MX模式）](./matmul-mx.md)。A、B、C及Bias的信息说明见下表：

**表** 矩阵信息说明

| 矩阵 | 存储位置 | 形状（行数×列数） | 数据格式 | 分形大小（行数×列数） |
| --- | --- | --- | --- | --- |
| A | L0A Buffer | M×K | Nz | 16×K0 |
| B | L0B Buffer | K×N | Zn | K0×16 |
| C | L0C Buffer | M×N | Nz | 16×16 |
| Bias（用于C矩阵初始化） | BiasTable Buffer | 1×N，使用时通过广播复制M行来初始化C矩阵 | ND | - |

表格中K0的取值为`32B / sizeof(dtype)`，`dtype`为矩阵的数据类型。

## 函数原型

```python
matmul(dst: Tensor, lhs: Tensor, rhs: Tensor, *, bias: Tensor | None = None, init = True, unit_flag: int = 0, disable_gemv: bool = True, mx_mode: bool = False) -> Tensor
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `dst` | 输入/输出 | 目的操作数，结果矩阵C位于L0C Buffer，起始地址需按照1024字节对齐。初始累加值由`init`和`bias`共同决定，见[初始化与累加](#初始化与累加)。数据类型取值请参见下文支持的数据类型组合。 |
| `lhs` | 输入 | 源操作数，左矩阵A位于L0A Buffer，起始地址需按照512字节对齐。数据类型取值请参见[支持的数据类型](#支持的数据类型)。 |
| `rhs` | 输入 | 源操作数，右矩阵B位于L0B Buffer，起始地址需按照512字节对齐。数据类型取值请参见[支持的数据类型](#支持的数据类型)。 |
| `bias` | 输入 | Bias矩阵在BiasTable Buffer中的起始地址。初始化时，将Bias沿M方向广播，作为初始累加值；不提供时，由`init`决定清零或使用`dst`原值，见[初始化与累加](#初始化与累加)。<br>Bias的数据类型需与C矩阵的数据类型保持一致。起始地址需按照64字节对齐，可以为非零地址。调用前，需通过`mem_copy`将Bias数据从L1 Buffer搬入。Bias占用长度为`N × dst的数据类型字节宽度`向上补齐到64字节。默认值为`None`。 |
| `init` | 输入 | 配置初始累加值，接受Python布尔值或运行时布尔标量，默认值为`True`。<br>&nbsp;&nbsp;&bull; `True`：重新初始化；未提供`bias`时初始值为0，提供时初始值为广播后的Bias。<br>&nbsp;&nbsp;&bull; `False`：不执行清零操作，矩阵C的初始值由`dst`中原有矩阵配置。<br>传入Python布尔值`False`时不能同时提供`bias`；运行时布尔标量的取值及与`bias`的组合见[初始化与累加](#初始化与累加)。 |
| `unit_flag` | 输入 | 用于控制矩阵乘加指令与矩阵搬出指令的细粒度并行，开启UnitFlag后，硬件每计算完一个分形，计算结果就会被搬出。取值说明如下：<br>&nbsp;&nbsp;&bull; `0`（DISABLE）：不开启UnitFlag。<br>&nbsp;&nbsp;&bull; `2`（ENABLE_KEEP）：开启UnitFlag，硬件执行完指令后不改变单元标志位。<br>&nbsp;&nbsp;&bull; `3`（ENABLE_UPDATE）：开启UnitFlag，硬件执行完指令后改变单元标志位。<br>矩阵乘加指令与对应的矩阵搬出指令必须都开启或都不开启UnitFlag，开启后指令之间无需再插入同步指令。<br>同步机制和迭代场景见[UnitFlag](./key-features-unit-flag.md)。默认值为`0`。 |
| `disable_gemv` | 输入 | M为1时，配置是否关闭GEMV模式。<br>&nbsp;&nbsp;&bull; false：开启GEMV模式。<br>&nbsp;&nbsp;&bull; true：关闭GEMV模式。<br>M不为1时，该参数不生效。<br>向量布局与使用场景见[GEMV](./key-features-gemv.md)。默认值为`True`。 |
| `mx_mode` | 输入 | 编译期Python布尔值，默认值为`False`。本页普通模式使用`False`；设置为`True`时执行带有量化系数的矩阵乘加，其功能、数据类型和量化系数矩阵的准备要求见[matmul（MX模式）](./matmul-mx.md)。 |

### 初始化与累加

`init`决定是否重新初始化累加器，`bias`指定重新初始化时使用的偏置数据。`bias`只能为`None`或Bias Tensor视图，不能取布尔值`False`。

`init`有以下两种传入方式：

- **Python布尔值**：在构建计算图时已经确定，例如直接传入`init=False`，或传入一个值为`False`的Python变量。这就是“编译期已确定的`False`”，不是设备执行时计算出的布尔结果。
- **运行时布尔标量**：在设备执行时确定取值，例如运行时循环索引与0比较的结果。其值为`True`时初始化，为`False`时继续累加。

两者与`bias`的完整组合如下。记$C_0$为L0C Buffer中原有的C矩阵，$\mathbf{1}_{M \times 1}$为全1列向量，$\mathrm{Bias}_{1 \times N}$为偏置矩阵。以$P$表示不包含初始累加值的矩阵乘法部分：

- 普通模式：$P = A \times B$。
- MX模式：$P = (\mathrm{ScaleA} \otimes A) \times (\mathrm{ScaleB} \otimes B)$，其中$\otimes$表示广播乘法，ScaleA、ScaleB分别为左右矩阵的量化系数矩阵。

| `init`的传入方式 | `init`的取值 | `bias` | 初始累加值 | 本次计算 |
| --- | --- | --- | --- | --- |
| Python布尔值 | `True` | `None` | 全0矩阵 | $C = P$ |
| Python布尔值 | `False` | `None` | L0C Buffer中原有的C矩阵 | $C = P + C_0$ |
| Python布尔值 | `True` | Bias Tensor视图 | 沿M方向广播后的Bias矩阵 | $C = P + \mathbf{1}_{M \times 1}\,\mathrm{Bias}_{1 \times N}$ |
| Python布尔值 | `False` | Bias Tensor视图 | 不支持 | — |
| 运行时布尔标量 | 运行时为`True` | `None` | 全0矩阵 | $C = P$ |
| 运行时布尔标量 | 运行时为`False` | `None` | L0C Buffer中原有的C矩阵 | $C = P + C_0$ |
| 运行时布尔标量 | 运行时为`True` | Bias Tensor视图 | 沿M方向广播后的Bias矩阵 | $C = P + \mathbf{1}_{M \times 1}\,\mathrm{Bias}_{1 \times N}$ |
| 运行时布尔标量 | 运行时为`False` | Bias Tensor视图 | L0C Buffer中原有的C矩阵，不读取Bias | $C = P + C_0$ |

例如，沿K轴分块计算时，可将运行时首块条件作为`init`并传入`bias`：首块使用Bias初始化，后续块在已有C上累加，不再次加入Bias。直接传入Python布尔值`init=False`并提供`bias`则会被编译器拒绝，因为该调用不会使用Bias；这是当前DSL的接口约束。

### 支持的数据类型

普通模式（`mx_mode=False`）下，`lhs`、`rhs`、`dst`、`bias`的数据类型取值组合见下表：

**表** 普通模式支持的数据类型组合

| lhs | rhs | dst | bias |
| --- | --- | --- | --- |
| `dtypes.int8` | `dtypes.int8` | `dtypes.int32` | `dtypes.int32` |
| `dtypes.hifloat8` | `dtypes.hifloat8` | `dtypes.float32` | `dtypes.float32` |
| `dtypes.float8_e5m2` | `dtypes.float8_e5m2` | `dtypes.float32` | `dtypes.float32` |
| `dtypes.float8_e5m2` | `dtypes.float8_e4m3fn` | `dtypes.float32` | `dtypes.float32` |
| `dtypes.float8_e4m3fn` | `dtypes.float8_e5m2` | `dtypes.float32` | `dtypes.float32` |
| `dtypes.float8_e4m3fn` | `dtypes.float8_e4m3fn` | `dtypes.float32` | `dtypes.float32` |
| `dtypes.float16` | `dtypes.float16` | `dtypes.float32` | `dtypes.float32` |
| `dtypes.bfloat16` | `dtypes.bfloat16` | `dtypes.float32` | `dtypes.float32` |
| `dtypes.float32` | `dtypes.float32` | `dtypes.float32` | `dtypes.float32` |

## 返回值说明

返回本次矩阵乘加结果的`Tensor`视图，与`dst`共享L0C Buffer中的存储，并包含本次计算的有效数据范围和布局信息。后续累加或搬运应使用该返回视图。当`init=False`且未提供`bias`时，在`dst`原有矩阵的基础上累加乘法结果。

## 约束说明

### 通用约束

- 本接口仅在AIC上生效。
- `M`、`K`、`N`中的任意一个值为0时，接口将被视为NOP（空操作）。

### 内存使用约束

- L0C Buffer大小为256KB，L0A Buffer和L0B Buffer大小均为64KB。BiasTable Buffer大小为4KB。矩阵的起始地址和占用空间不能超出对应Buffer的范围。
- 申请矩阵存储空间时，需使用按照分形大小补齐后的数值进行申请：M、N分别向上补齐到16的倍数，K向上补齐到K0的倍数，K0的取值为`32B / sizeof(dtype)`，`dtype`为矩阵的数据类型。`lhs`、`rhs`、`dst`的shape即为矩阵的有效M、K、N值，补齐部分为无效数据，不参与结果矩阵有效区域的计算。
- M、K、N的单位为元素，取值范围均为[0, 4095]。
- 当M为1且`disable_gemv`为false时，将开启GEMV模式。此时从L0A Buffer读取矩阵A时按照ND格式读取，矩阵A需按照ND格式排布，起始地址仍需按照512字节对齐。

### UnitFlag约束

- 开启UnitFlag时，矩阵乘加指令与对应矩阵搬出指令需同时开启UnitFlag。当希望同一块L0C Buffer内存空间能持续只被多条矩阵乘加指令或多条矩阵搬出指令操作时，除最后一条外的指令需将`unit_flag`设置为`2`（ENABLE_KEEP），维持被操作内存空间的持续占用状态，最后一条指令设置为`3`（ENABLE_UPDATE），解除被占用状态。
- 开启UnitFlag时，矩阵计算方向需与矩阵搬出读取顺序保持一致。矩阵搬出指令开启Nz2ND随路格式转换，或未进行随路格式转换但开启B8/B4量化并触发Channel Merge功能时，调用`cube.set_mmad_direction("n")`；其他场景调用`cube.set_mmad_direction("m")`。
- 开启UnitFlag时，建议矩阵乘加的计算数据量与矩阵搬出的数据量保持一致。两者不一致可能导致执行异常。

### 特殊值/边界值约束

- 注意，应避免nan输入，否则可能会产生执行报错；整数类型仅支持饱和模式。

## 调用示例

以下示例演示`matmul`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, copy_options, dtypes, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops import cube
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.tensor import MemLoc

@kernel
def _matmul_kernel(a, b, bias, c):
    l1a = Buffer(MemLoc.L1, dtypes.float16, make_layout((16, 16)))
    l1b = Buffer(MemLoc.L1, dtypes.float16, make_layout((16, 16)))
    bias_l1 = Buffer(MemLoc.L1, dtypes.float32, make_layout((16,)), data_format="nd")
    bias_bt = Buffer(MemLoc.BIAS, dtypes.float32, make_layout((16,)))
    l0a = Buffer(MemLoc.L0A, dtypes.float16, make_layout((16, 16)))
    l0b = Buffer(MemLoc.L0B, dtypes.float16, make_layout((16, 16)))
    l0c = Buffer(MemLoc.L0C, dtypes.float32, make_layout((16, 16)))
    mem_copy(l1a, a)
    mem_copy(l1b, b)
    mem_copy(bias_l1, bias)
    mem_copy(bias_bt, bias_l1)
    mem_copy(l0a, l1a)
    mem_copy(l0b, l1b)
    cube.set_mmad_direction("n")
    result = matmul(
        l0c, l0a, l0b,
        bias=bias_bt,
        init=True,
        unit_flag=3,
        disable_gemv=True,
        mx_mode=False,
    )
    mem_copy(c, result, options=copy_options(unit_flag=3))

@host
def run(a, b, bias, c):
    _matmul_kernel[1](a, b, bias, c)

def main():
    a = torch.arange(256, dtype=torch.float16, device="npu:0").reshape(16, 16) / 16.0
    b = a.clone()
    bias = torch.arange(16, dtype=torch.float32, device="npu:0") + 1.0
    c = torch.empty((16, 16), dtype=torch.float32, device="npu:0")
    run(a, b, bias, c)
    torch.npu.synchronize()

    expected = a.cpu().float() @ b.cpu().float().T + bias.cpu().reshape(1, 16)
    torch.testing.assert_close(c.cpu(), expected, rtol=1e-2, atol=1e-2)
    print(f"first={c.cpu()[0, 0].item():.3f}, last={c.cpu()[-1, -1].item():.3f}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
first=5.844, last=3845.844
```
