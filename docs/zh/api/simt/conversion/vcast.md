# simt.vcast

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将短向量value数值转换为dtype。舍入与饱和效果取决于具体类型和参数组合。

## 函数原型

```python
simt.vcast(value, *, dtype, rounding=RoundingMode.RN, saturate=False) -> 短向量
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| value | 输入 | 双分量短向量，类型为`float32x2`、`float16x2`、`bfloat16x2`、`fp8x2_e4m3fn`、`fp8x2_e5m2`、`hifloat8x2`；具体源、目标组合见约束说明。 |
| dtype | 输入 | 目标双分量短向量类型描述符，可选`float32x2`、`float16x2`、`bfloat16x2`、`fp8x2_e4m3fn`、`fp8x2_e5m2`、`hifloat8x2`；必须与源类型不同。 |
| rounding | 输入 | [RoundingMode](./roundingmode.md) 枚举或名称字符串，默认`RoundingMode.RN`；是否生效取决于源、目标类型。 |
| saturate | 输入 | Python bool，默认False，是否请求对应转换的饱和形式。 |

## 返回值说明

返回dtype的新值，不修改输入。

## 约束说明

仅使用下表中的双分量短向量组合；源和目标数据类型必须不同，不进行分量广播。

| 源类型 | 目标类型 | 额外限制 |
| --- | --- | --- |
| `float32x2` | `float16x2`、`bfloat16x2` | 舍入和饱和选项取决于目标类型。 |
| `float16x2`、`bfloat16x2` | `float32x2` | `rounding`、`saturate`不生效。 |
| `float32x2` | `fp8x2_e4m3fn`、`fp8x2_e5m2` | 不能通过`rounding`改变舍入规则。 |
| `fp8x2_e4m3fn`、`fp8x2_e5m2` | `float32x2` | `rounding`、`saturate`不生效。 |
| `float32x2`、`float16x2` | `hifloat8x2` | `RH`使用对应舍入规则，其他模式按`RNA`处理。 |
| `hifloat8x2` | `float32x2`、`float16x2` | `rounding`、`saturate`不生效。 |

- 不能把上表理解为所有列出类型都能两两互转；例如未列出`float16x2`与`bfloat16x2`的直接转换。
- 不使用整数短向量或三、四分量短向量替代上述类型。
- `saturate=True`不是通用的`[0, 1]`截断；半精度饱和转换还依赖设备配置。

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
