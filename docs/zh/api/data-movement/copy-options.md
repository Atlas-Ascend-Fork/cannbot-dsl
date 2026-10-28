# copy_options
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

创建不可变的数据搬运配置，传给 `mem_copy(dst, src, options=...)`，用于选择搬运方式、目标 Vector 子核、缓存策略及原子累加等行为；本接口不搬运数据。
传入已有配置 `base` 时，只覆盖显式指定的选项，其余选项继承自 `base`，且不修改 `base`。
与本接口不同，Tensor 的布局及 `transpose`、`pad`、`convert`、`relu` 等视图或数据变换接口描述要搬运的数据及其变换，实际搬运由 `mem_copy` 执行。

## 函数原型

```python
copy_options(
    base: CopyOptions | None = None,
    *,
    engine: str | None = ...,
    dst_subblock: int | ScalarValue | None = ...,
    unit_flag: int | ScalarValue = ...,
    mx_scale: Tensor | None = ...,
    l2_cache_ctl: int | ScalarValue = ...,
    atomic_add: bool | ScalarValue = ...,
    channel_split: bool | None = ...,
) -> CopyOptions
```

调用入口为 `cannbotdsl.copy_options`。除 `base` 外，所有参数均须以关键字传入。
原型中的 `...` 表示“省略该选项”，不是可传入的参数值：有 `base` 时继承其对应选项，没有 `base` 时采用下表所列默认值。显式传入 `None` 不表示继承，仅可用于类型允许 `None` 的选项。
`ScalarValue` 表示 DSL 标量，具体数据类型要求见参数说明。

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `base` | 输入 | 用作派生基础的 [CopyOptions](copy-options-type.md) 对象。默认 `None`，表示从默认配置创建。不能传入字典或其他自定义配置对象。 |
| `engine` | 输入 | 搬运方式，须为编译期值。默认 `"auto"`；`None` 与 `"auto"` 等价。`"auto"` 使用普通 DMA，**不会自动选择 NDDMA**。`"nddma"` 显式选择 NDDMA，仅用于 GM→UB 的同数据类型、不做格式转换的搬运，支持 1～5 维视图；需要边界填充时，通过 `pad` 构造源视图，不在本接口中设置填充方式。 |
| `dst_subblock` | 输入 | 指定接收数据的 Vector 子核编号，取值为 `0` 或 `1`，支持 Python 整数常量或 `int32` 类型的 DSL 标量。仅用于单目标 L0C→UB、L1→UB 搬运，不能与双目标分区搬运组合。默认 `None`，不显式指定，单目标搬运默认选择子核 `0`。 |
| `unit_flag` | 输入 | L0C 搬出操作的 unit flag 控制值，用于计算与搬运之间的标志协同。默认 `0`，表示不启用该控制；接口接收 `0`～`3` 的 Python 整数常量或 `int32` 类型的 DSL 标量。非 L0C 搬出方向必须保持常量 `0`。它不代替数据依赖所需的同步。 |
| `mx_scale` | 输入 | 与 L1 数据一起加载到 L0A/L0B 的 MX 缩放因子 [Tensor](../data-structures/tensor/overview.md)，默认 `None`，不加载缩放因子。必须位于 L1，数据类型为 `fp8_e8m0`，数据与缩放因子均为二维视图；目标数据类型为 `fp8_e4m3fn`、`fp8_e5m2` 或 `fp4x2_e2m1`。加载到 L0A 时缩放因子使用 ZN 布局，加载到 L0B 时使用 NZ 布局；其外轴长度须与数据外轴一致，沿归约轴每 64 个数据元素对应 2 个缩放因子，不足 64 个按一组计。 |
| `l2_cache_ctl` | 输入 | 本次 GM 访问的 L2 缓存策略编码，默认 `0`；支持 Python 整数常量或 `int32` 类型的 DSL 标量。GM→UB（含 NDDMA）可取 `0`、`1`、`2`、`4`、`5`、`6`；UB→GM、GM→L1、L0C→GM 可取 `0`、`1`、`2`、`4`；其他方向必须保持常量 `0`。GM→UB 的 `0`、`1`、`2` 分别对应普通缓存的优先替换、最后替换、持久保留策略，`4`、`5`、`6` 分别对应不分配缓存行并保留、清理、丢弃命中行的策略；写 GM 的 `4` 对应不分配并清理，不能按读侧的 `4` 解释。运行时值也须满足对应方向的取值范围。 |
| `atomic_add` | 输入 | 是否将搬运数据原子累加到 GM 目标的原值上。默认 `False`，表示覆盖写入；支持 Python 布尔常量或布尔类型的 DSL 标量。启用时仅支持 UB→GM、L0C→GM，GM 目标数据类型可为 `int8`、`int16`、`int32`、`bfloat16`、`float16`、`float32`，并须满足该搬运方向的数据类型要求。L0C→GM 的 NZ 直存形式不支持原子累加。 |
| `channel_split` | 输入 | L0C→GM/UB/L1 搬运的 FP32 通道拆分方式，须为编译期 `bool` 或 `None`，默认 `None`（自动）。显式指定 `True` 或 `False` 时，源、目标都必须为 `float32`；GM/UB 目标须使用 NZ 布局。`True` 将源块拆分为内轴长度 C0 为 8 的目标块，`False` 保持 C0 为 16 的源块组织，目标必须预先声明相匹配的物理布局；本选项不会修改目标布局。`True` 不能与 L0C→UB 的双目标分区搬运组合。 |

GM、UB、NZ、ZN、C0 等术语见[术语表](../../guide/glossary.md)。

例如，`copy_options(base, l2_cache_ctl=2)` 只修改缓存策略；若 `base` 使用 NDDMA，派生配置仍使用 NDDMA。需要恢复普通 DMA 时，须显式指定 `engine="auto"` 或 `engine=None`，不能只省略 `engine`。

## 返回值说明

返回新的 [CopyOptions](copy-options-type.md) 对象，作为一次或多次 `mem_copy` 调用的 `options` 参数使用。返回值不是 Tensor，不分配数据缓冲区，也不保存一份源数据副本。

## 约束说明

- 配置引用的运行时标量和 Tensor 必须在使用该配置的 `mem_copy` 调用处有效；不能通过复用配置，将一个 Kernel 的局部操作数带入另一个独立 Kernel。
- 配置不承担同步职责。使用同一片上缓冲区进行连续读写时，应通过 `RingBuffer` 的 `next_write()`、`next_read()` 选择写入与读取的槽位，使数据访问具有正确的先后依赖；不能在前一次读取完成前覆盖其数据。
- 原子累加会读取 GM 目标的原值。首次累加前须初始化该区域，并避免在累加期间对同一区域进行未同步的普通写入。

## 调用示例

以下示例显式选择 NDDMA，将一个 `(16, 8)` 的 `float16` Tensor 的转置视图搬入 UB，再写回 GM，并在 Host 侧校验结果。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
import cannbotdsl as cb
import torch
import torch_npu


@cb.kernel
def transpose_copy_kernel(src: cb.Tensor, dst: cb.Tensor):
    slots = cb.RingBuffer(
        cb.MemLoc.UB, shape=(8, 16), dtype=cb.dtypes.float16, depth=1
    )
    options = cb.copy_options(engine="nddma")
    cb.mem_copy(slots.next_write(), src.transpose(0, 1), options=options)
    cb.mem_copy(dst, slots.next_read())


@cb.host
def transpose_copy(src: cb.Tensor, dst: cb.Tensor):
    transpose_copy_kernel[1](src, dst)


def main():
    src_cpu = torch.arange(128, dtype=torch.float16).reshape(16, 8)
    src = src_cpu.to("npu:0")
    dst = torch.empty((8, 16), dtype=torch.float16, device="npu:0")
    transpose_copy(src, dst)
    torch.npu.synchronize()
    torch.testing.assert_close(dst.cpu(), src_cpu.t(), rtol=0, atol=0)
    print("copy_options example passed")


if __name__ == "__main__":
    main()
```

预期结果：输出 Tensor 的形状为 `(8, 16)`，元素与输入 Tensor 的转置逐元素一致；校验通过后打印 `copy_options example passed`。
