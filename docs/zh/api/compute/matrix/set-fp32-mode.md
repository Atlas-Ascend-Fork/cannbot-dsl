# set_fp32_mode
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

本接口用于设置Mmad计算关闭HF32模式，其作用与`enable_hf32`相反，两个接口不同时生效。关闭HF32模式后，L0A Buffer与L0B Buffer中的`dtypes.float32`数据在参与Mmad计算之前不做舍入处理。

本接口为矩阵计算相关配置接口，仅在AIC上生效。

## 函数原型

```python
set_fp32_mode() -> None
```

## 参数说明

无。

## 返回值说明

无返回值。

## 约束说明

- 本接口运行在`[PIPE](../../execution/pipeline.md).S`流水上。
- 本接口非AIC调用直接返回。
- 本接口需在矩阵乘加指令（`matmul`）执行前调用，以此来确保模式配置在矩阵乘加计算过程中生效。
- 与`enable_hf32`作用相反，二者不同时生效。
- FP32模式启用后会持续生效，不会自动关闭。后续矩阵乘加指令若不显式重新配置，将沿用当前模式。如需开启HF32模式，请重新调用`enable_hf32`接口。

## 调用示例

以下示例演示`set_fp32_mode`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, dtypes, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.cube import enable_hf32, set_fp32_mode
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.tensor import MemLoc

# HF32 的尾数为 10 bit，1.0 与 1 + 2^-10 的中点即 1 + 2^-11，恰好落在舍入的平局点上。
TIE = 1.0 + 0.5 / 1024.0

@kernel
def _set_fp32_mode_kernel(a, b, c):
    l1a = Buffer(MemLoc.L1, dtypes.float32, make_layout((16, 16)))
    l1b = Buffer(MemLoc.L1, dtypes.float32, make_layout((16, 16)))
    l0a = Buffer(MemLoc.L0A, dtypes.float32, make_layout((16, 16)))
    l0b = Buffer(MemLoc.L0B, dtypes.float32, make_layout((16, 16)))
    l0c = Buffer(MemLoc.L0C, dtypes.float32, make_layout((16, 16)))
    mem_copy(l1a, a)
    mem_copy(l1b, b)
    enable_hf32()
    set_fp32_mode()
    mem_copy(l0a, l1a)
    mem_copy(l0b, l1b)
    result = matmul(l0c, l0a, l0b, init=True)
    mem_copy(c, result)

@host
def run(a, b, c):
    _set_fp32_mode_kernel[1](a, b, c)

def main():
    a = torch.eye(16, dtype=torch.float32, device="npu:0") * TIE
    b = torch.eye(16, dtype=torch.float32, device="npu:0")
    c = torch.empty((16, 16), dtype=torch.float32, device="npu:0")
    run(a, b, c)
    torch.npu.synchronize()

    # 先开启HF32模式再关闭，L0A Buffer/L0B Buffer中的float数据不再做舍入处理，
    # 结果即为TIE本身；若关闭未生效，TIE将被舍入为1.0。
    torch.testing.assert_close(c.cpu(), torch.eye(16, dtype=torch.float32) * TIE)
    print(f"tie={a.cpu()[0, 0].item():.8f} -> fp32={c.cpu()[0, 0].item():.8f}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
tie=1.00048828 -> fp32=1.00048828
```
