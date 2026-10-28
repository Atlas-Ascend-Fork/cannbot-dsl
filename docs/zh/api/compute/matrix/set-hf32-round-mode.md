# set_hf32_round_mode
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

本接口用于设置HF32模式舍入方式，使用该接口前需要先使用`enable_hf32`开启HF32模式。

## 函数原型

```python
set_hf32_round_mode(mode: HF32RoundingMode) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `mode` | 输入 | FP32输入转换为HF32时的舍入方式，类型为`HF32RoundingMode`，必选，不接受原始整数或字符串。应在开启HF32后、执行矩阵乘加前配置。 |

## 返回值说明

无返回值。

## 约束说明

- 本接口运行在`[PIPE](../../execution/pipeline.md).S`流水上。
- 首次开启HF32且此前未配置舍入模式时，默认使用`NEAREST_EVEN`；关闭后再次开启时保留已有舍入配置。
- 本接口需在矩阵乘加指令（`matmul`）执行前调用，以此来确保模式配置在矩阵乘加计算过程中生效。
- 本接口配置的舍入模式仅在HF32模式开启期间生效，需先调用`enable_hf32`开启HF32模式再调用本接口，否则会报错。
- 舍入模式配置后会持续生效，HF32模式关闭后再次开启仍将沿用上次的舍入模式配置，如需切换舍入模式，请重新调用本接口。

## 调用示例

以下示例演示`set_hf32_round_mode`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, dtypes, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.cube import HF32RoundingMode, enable_hf32, set_hf32_round_mode
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.tensor import MemLoc

# HF32 的尾数为 10 bit，1.0 与 1 + 2^-10 的中点即 1 + 2^-11，恰好落在舍入的平局点上。
TIE = 1.0 + 0.5 / 1024.0

@kernel
def _set_hf32_round_mode_kernel(a, b, c):
    l1a = Buffer(MemLoc.L1, dtypes.float32, make_layout((16, 16)))
    l1b = Buffer(MemLoc.L1, dtypes.float32, make_layout((16, 16)))
    l0a = Buffer(MemLoc.L0A, dtypes.float32, make_layout((16, 16)))
    l0b = Buffer(MemLoc.L0B, dtypes.float32, make_layout((16, 16)))
    l0c = Buffer(MemLoc.L0C, dtypes.float32, make_layout((16, 16)))
    mem_copy(l1a, a)
    mem_copy(l1b, b)
    enable_hf32()
    set_hf32_round_mode(HF32RoundingMode.NEAREST_AWAY)
    mem_copy(l0a, l1a)
    mem_copy(l0b, l1b)
    result = matmul(l0c, l0a, l0b, init=True)
    mem_copy(c, result)

@host
def run(a, b, c):
    _set_hf32_round_mode_kernel[1](a, b, c)

def main():
    a = torch.eye(16, dtype=torch.float32, device="npu:0") * TIE
    b = torch.eye(16, dtype=torch.float32, device="npu:0")
    c = torch.empty((16, 16), dtype=torch.float32, device="npu:0")
    run(a, b, c)
    torch.npu.synchronize()

    # NEAREST_AWAY将平局值TIE舍入到远离零的一侧，即1 + 2^-10；若使用NEAREST_EVEN则舍入为1.0。
    torch.testing.assert_close(c.cpu(), torch.eye(16, dtype=torch.float32) * (1.0 + 1.0 / 1024.0))
    print(f"tie={a.cpu()[0, 0].item():.8f} -> hf32={c.cpu()[0, 0].item():.8f}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
tie=1.00048828 -> hf32=1.00097656
```
