# simt.vcast

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将双分量短向量`value`转换为`dtype`指定的数据类型。支持的类型组合、舍入模式和饱和选项见[约束说明](#约束说明)。

## 函数原型

```python
simt.vcast(value, *, dtype, rounding=RoundingMode.RN, saturate=False) -> 短向量
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| value | 输入 | 双分量短向量，类型为`float32x2`、`float16x2`、`bfloat16x2`、`fp8x2_e4m3fn`、`fp8x2_e5m2`、`hifloat8x2`；具体源、目标组合见约束说明。 |
| dtype | 输入 | 目标双分量短向量数据类型，可选`float32x2`、`float16x2`、`bfloat16x2`、`fp8x2_e4m3fn`、`fp8x2_e5m2`、`hifloat8x2`；必须与源类型不同。 |
| rounding | 输入 | [RoundingMode](./roundingmode.md)枚举或名称字符串，默认`RoundingMode.RN`。各类型组合的可选值见[约束说明](#约束说明)。 |
| saturate | 输入 | 是否启用饱和转换，类型为`bool`，默认`False`。各类型组合的支持情况和`CTRL[60]`设置要求见[约束说明](#约束说明)。 |

## 返回值说明

返回dtype的新值，不修改输入。

## 约束说明

源和目标须为下表中的双分量短向量，且类型不同。不支持整数短向量、三分量或四分量短向量，也不进行分量广播。

| 源类型 | 目标类型 | `rounding` | `saturate` |
| --- | --- | --- | --- |
| `float32x2` | `float16x2` | `RN`、`RNA`、`RD`、`RU`、`RZ`、`RO`。 | `False`或`True`；启用时须满足下文`CTRL[60]`设置要求。 |
| `float32x2` | `bfloat16x2` | `RN`、`RNA`、`RD`、`RU`、`RZ`。 | `False`或`True`；启用时须满足下文`CTRL[60]`设置要求。 |
| `float16x2`、`bfloat16x2` | `float32x2` | 不生效。 | 不生效。 |
| `float32x2` | `fp8x2_e4m3fn`、`fp8x2_e5m2` | 固定为`RN`，`rounding`参数不生效。 | `False`或`True`；启用时须满足下文`CTRL[60]`设置要求。 |
| `fp8x2_e4m3fn`、`fp8x2_e5m2` | `float32x2` | 不生效。 | 不生效。 |
| `float32x2`、`float16x2` | `hifloat8x2` | `RH`使用混合舍入；`RN`、`RNA`、`RD`、`RU`、`RZ`、`RO`均按`RNA`处理。 | `False`或`True`；启用时须满足下文`CTRL[60]`设置要求。 |
| `hifloat8x2` | `float32x2`、`float16x2` | 不生效。 | 不生效。 |

- 不支持表外类型组合，包括`float16x2`与`bfloat16x2`直接互转。
- 涉及`fp8x2_e4m3fn`、`fp8x2_e5m2`或`hifloat8x2`的转换，仅用于SIMD与SIMT混合执行场景。
- 表中标明支持`saturate=True`的转换受`CTRL[60]`控制：须在进入SIMT函数前，在SIMD侧将`CTRL[60]`置为`0`。纯SIMT Kernel中不能通过该参数启用这些转换的饱和模式。寄存器设置接口见[set_ctrl](../../execution/device/spr/set-ctrl.md)。

## 调用示例

将两分量浮点短向量转为半精度，再转回单精度并打印分量。

```python
import torch
import torch_npu
from cannbotdsl import simt, dtypes, dim3
from cannbotdsl.lang.host import host
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.tensor import Tensor


@kernel
def demo(src: Tensor, dst: Tensor):
    with vf(mode='simt'):
        x = simt.make_short_vector(src[0], src[1])
        y = simt.vcast(x, dtype=dtypes.float16x2)
        z = simt.vcast(y, dtype=dtypes.float32x2)
        dst[0] = z.x
        dst[1] = z.y


@host
def run(src, dst):
    demo[dim3(1), dim3(1)](src, dst)


def main():
    src = torch.tensor([1.5, 2.5, 3.5, 4.5], device='npu', dtype=torch.float32)
    dst = torch.zeros(2, device='npu', dtype=torch.float32)
    run(src, dst)
    torch.npu.synchronize()
    assert dst.cpu().tolist() == [1.5, 2.5]
    print(dst.cpu().tolist())


if __name__ == "__main__":
    main()
```

预期结果：

```text
[1.5, 2.5]
```
