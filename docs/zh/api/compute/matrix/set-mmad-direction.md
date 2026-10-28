# set_mmad_direction
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

本接口用于设置矩阵乘加计算（Mmad）时逐列或逐行生成矩阵计算结果分形：`'m'`为逐列生成，`'n'`为逐行生成。该配置只改变L0C Buffer上结果分形的生成顺序，不改变计算结果、乘加次数或输出格式。

在开启UnitFlag的场景下，当L0C Buffer上结果分形的生成顺序与数据搬出指令的搬出顺序保持一致时，可以获得更好的性能表现，因此本接口可用于以下场景：

- `'m'`（逐列生成矩阵计算结果分形）：
  - UnitFlag关闭时作为所有搬出场景的默认配置（此时不同的结果生成方向通常没有实质性能差异）。
  - UnitFlag开启且搬出时不进行随路格式转换，且未因开启B8/B4量化触发Channel Merge功能。
  - UnitFlag开启且搬出时开启Nz2DN随路格式转换。
- `'n'`（逐行生成矩阵计算结果分形）：
  - UnitFlag开启且搬出时开启Nz2ND随路格式转换。
  - UnitFlag开启且搬出时不进行随路格式转换，且开启B8/B4量化触发Channel Merge功能。

## 函数原型

```python
set_mmad_direction(direction: Literal['m', 'n']) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `direction` | 输入 | 设置矩阵乘加（Mmad）计算时的结果分形生成方向。<br>&bull; `'n'`：逐行生成矩阵计算结果分形，与开启Nz2ND随路格式转换或B8/B4量化Channel Merge的矩阵搬出指令配合使用。<br>&bull; `'m'`：逐列生成矩阵计算结果分形，与其他矩阵搬出场景配合使用。 |

## 返回值说明

无返回值。

## 约束说明

- 本接口运行在`[PIPE](../../execution/pipeline.md).S`流水上。
- 本接口需在矩阵乘加指令（`matmul`）执行前调用，以此来确保模式配置在矩阵乘加计算过程中生效。
- 方向配置一旦写入会持续生效，后续矩阵乘加指令若不显式重新配置，将沿用当前方向配置。如需切换为逐行生成矩阵计算结果分形，请重新调用`set_mmad_direction("n")`接口。

## 调用示例

以下示例演示`set_mmad_direction`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, dtypes, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.cube import set_mmad_direction
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.tensor import MemLoc

@kernel
def _set_mmad_direction_kernel(a, b, c):
    l1a = Buffer(MemLoc.L1, dtypes.float16, make_layout((16, 16)))
    l1b = Buffer(MemLoc.L1, dtypes.float16, make_layout((16, 16)))
    l0a = Buffer(MemLoc.L0A, dtypes.float16, make_layout((16, 16)))
    l0b = Buffer(MemLoc.L0B, dtypes.float16, make_layout((16, 16)))
    l0c = Buffer(MemLoc.L0C, dtypes.float32, make_layout((16, 16)))
    mem_copy(l1a, a)
    mem_copy(l1b, b)
    set_mmad_direction("m")
    mem_copy(l0a, l1a)
    mem_copy(l0b, l1b)
    result = matmul(l0c, l0a, l0b, init=True)
    mem_copy(c, result)

@host
def run(a, b, c):
    _set_mmad_direction_kernel[1](a, b, c)

def main():
    a = torch.arange(256, dtype=torch.float16, device="npu:0").reshape(16, 16) / 16.0
    b = a.clone()
    c = torch.empty((16, 16), dtype=torch.float32, device="npu:0")
    run(a, b, c)
    torch.npu.synchronize()

    # 本接口只改变L0C Buffer上结果分形的生成顺序，不改变计算结果，此处以一次矩阵乘加验证配置调用已生效。
    torch.testing.assert_close(c.cpu(), a.cpu().float() @ b.cpu().float().T, rtol=1e-2, atol=1e-2)
    print(f"first={c.cpu()[0, 0].item():.3f}, last={c.cpu()[-1, -1].item():.3f}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
first=4.844, last=3829.844
```
