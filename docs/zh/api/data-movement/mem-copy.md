# mem_copy
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

在 Kernel 中将 `src` 描述的数据搬运到 `dst` 指定的存储区域，默认覆盖写入，启用原子累加时与目标原值相加。
Tensor 的布局及 `transpose`、`pad`、`convert`、`relu` 等视图或数据变换接口描述数据的访问方式和变换，[copy_options](./copy-options.md) 提供搬运控制选项，`mem_copy` 执行实际搬运，不返回目标 Tensor 视图。
与只创建视图的操作不同，本接口会读写存储；创建源视图时不复制数据，源数据在搬运执行时读取。

## 函数原型

```python
mem_copy(
    dst: Tensor | Storage | tuple[Tensor, ...] | list[Tensor],
    src: Tensor | Storage | tuple[Tensor, ...] | list[Tensor],
    options: CopyOptions | None = None,
) -> None
```

调用入口为 `cannbotdsl.mem_copy`。`options` 可按位置传入，也可写成 `options=...`。原型保留源码类型注解；其中裸 `Storage` 作为源的用法当前不支持，须先取得 Tensor 视图，详见参数说明。

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `dst` | 输入/输出 | 搬运目标。单目标接受 [Tensor](../data-structures/tensor/overview.md) 或 [Storage](../data-structures/storage/storage/overview.md)：Tensor 保持已声明的地址映射，Storage 按其布局策略和本次源数据范围生成目标视图，已有容量须足够容纳结果，不会自动扩容。非空 tuple/list 用于多目标搬运，元素须为 Tensor，规则见下文。默认仅写入目标；启用 `options` 中的 `atomic_add` 时还会读取目标原值。 |
| `src` | 输入 | 搬运源。接受 Tensor，或用于多源搬运的非空 Tensor tuple/list；裸 Storage 不可直接传入，应使用已有的 Tensor，或调用 `Storage.view()` 取得明确的源视图。单目标调用要求源、目标视图的逻辑维数一致，不会按元素总数自动展平或重塑；需要改变维数时应先显式构造视图。普通逻辑搬运的范围由源视图决定，目标须覆盖该范围；源可通过受支持的视图或变换接口描述转置、填充、拼接等行为。L1→UB 的物理搬运规则见下表。 |
| `options` | 输入 | 可选的 [CopyOptions](copy-options-type.md) 对象，由 [copy_options](./copy-options.md) 创建；默认 `None`，使用默认控制项。GM→UB 默认使用普通 DMA，不会自动选择 NDDMA；需要 NDDMA 时显式传入 `copy_options(engine="nddma")`。选项含义、取值及适用方向统一见 [copy_options 参数说明](./copy-options.md#参数说明)，不直接向 `mem_copy` 传入 `engine`、`atomic_add`、`mx_scale` 等旧参数。 |

源、目标的存储位置决定可使用的搬运方向。下表说明各方向的主要行为及对应操作数要求，不表示任意数据类型、布局和变换均可组合。GM、UB、NZ、ZN 等术语见[术语表](../../guide/glossary.md)。

| 源→目标 | 搬运行为 | 操作数及配置要求 |
| --- | --- | --- |
| GM→UB | 普通搬运，或显式选择 NDDMA | 源、目标数据类型一致。NDDMA 用于 1～5 维搬运，具体要求见 `copy_options` 的 `engine` 参数；源非连续并不意味着自动启用 NDDMA。 |
| GM→L1 | 普通搬运，或按源、目标布局完成 ND→NZ、DN→NZ 转换 | 格式转换由 Tensor 布局确定，不通过 `format_transform` 参数指定。 |
| UB→GM | 普通搬运，可启用原子累加 | 原子累加的数据类型及配置要求见 `copy_options` 的 `atomic_add` 参数。 |
| UB→UB | 相同布局的普通搬运 | 不支持通过本接口执行 UB→UB 格式转换。 |
| UB→L1 | 普通搬运或 ND→NZ 转换 | 单源可按目标布局转换；多源只支持 ND L1 目标，规则见下文。 |
| L1→L0A/L0B | 数据加载、转置加载及配套 MX 缩放因子加载 | 转置由 `src.transpose(0, 1)` 与目标布局共同表达；配套缩放因子通过 `copy_options(mx_scale=...)` 指定，不单独向 L0A/L0B 的缩放因子存储区搬运。 |
| L1→BIAS | Bias 数据加载 | 仅支持连续的一维视图，数据类型为 `float16`/`bfloat16`/`float32`→`float32` 或 `int32`→`int32`。目标逻辑容量须覆盖源，源物理容量须覆盖按 32 字节块向上对齐的读取量，32 位源须按两个块对齐；目标物理容量按 16 个元素向上对齐，对齐后的输出量不得超过 4096 字节。 |
| L1→UB | 从源起点按目标物理跨度复制字节 | 数据类型一致、物理布局静态；目标物理跨度须为正的 32 字节整数倍、不超过源跨度，且不超过 `65535 × 32` 字节。省略 `options`，或仅使用 `copy_options(dst_subblock=...)` 选择接收子核；不要传空的 `copy_options()`，也不要显式设置 `engine`、`channel_split` 或附加变换。 |
| L0C→GM/UB/L1 | 结果搬出，可组合受支持的类型转换、缩放和 ReLU；写 GM 时可启用原子累加 | 类型转换通过 `convert(src, dtype, scale=...)` 描述，转换结果的数据类型须与目标一致；ReLU 使用 `relu(src)`。GM/UB 的 NZ 目标仅支持同类型 `float32` 的 NZ 数据直接搬出，不能组合类型转换、缩放、ReLU 或原子累加；通道拆分要求见 `copy_options` 的 `channel_split` 参数。 |

多源搬运的 `src` tuple/list 等价于按输入顺序构造 `cat(src, dim=0)`，再搬运到一个目标；它不是任意地址的 gather：

- 各源须位于同一存储空间，均为同数据类型、同维数的 ND Tensor，除第 0 维外其余维度长度一致。GM 源可写入 ND UB、ND/NZ L1；UB 源仅可写入 ND L1。
- 源地址须具有可证明的共同基址和等间隔关系，访问须组成规则的搬运区间。不能用序列表示无规律的逐行索引；这类访问应分别调用 `mem_copy`。
- 源、目标数据类型一致且元素按整字节存储；ND→NZ 多源搬运仅支持 8、16、32 位元素。仅支持默认搬运控制和 `l2_cache_ctl`；UB→L1 的 `l2_cache_ctl` 必须为常量 `0`。
- UB→L1 多源搬运的地址、每段长度及段间间隔须按 32 字节对齐；最多 4095 段，每段长度为 1～65535 个 32 字节块，源、目标段间间隔为 0～65535 个块。运行时值也须满足这些范围。

多目标搬运的 `dst` tuple/list 将同一源数据复制到每个目标，不是对 Tensor 维度进行广播：

- 多于一个目标时，源须为单个 ND GM Tensor；目标须全部为 UB 或全部为 L1，且数据类型、逻辑布局和物理布局一致。目标支持 ND，L1 目标还支持 NZ；源与目标数据类型一致，元素按整字节存储，ND→NZ 时限于 8、16、32 位元素。
- 目标个数和顺序须在编译期确定。目标起始地址按列表顺序等间隔排列；调用者须保证第三个及之后的目标延续前两个目标确定的地址间隔。
- 仅支持默认搬运控制和 `l2_cache_ctl`，不组合填充、分区、原子累加或缩放。仅含一个 Tensor 的目标序列使用普通单目标规则。

## 返回值说明

无返回值，Python调用返回`None`。数据直接写入`dst`指定的存储，不另外分配一份数据副本。

后续读取或计算应使用事先构造的目标Tensor视图；需要从Storage取得操作数时，应显式调用`Storage.view()`绑定所需的数据范围和布局。不要将`mem_copy(...)`的返回值作为后续接口的Tensor入参。

## 约束说明

- 本接口用于设备侧 Kernel 搬运，不用于直接完成 CPU 与 NPU 之间的数据传输；GM→GM 也不支持直接调用本接口，应经受支持的片上存储中转。
- 多源拼接与多目标广播不能在同一调用中组合。需要“多源到多目标”时，应拆成分别满足上述规则的调用。
- 读写同一片上缓冲区时，必须保证写入完成后再读取，读取完成后再覆盖。使用 RingBuffer 时，通过同一缓冲区的 `next_write()`、`next_read()` 选择相应槽位并建立读写依赖；槽位选择本身不搬运数据。绕过缓冲区管理的跨流水或跨核访问须自行建立同步，不能只依赖 Python 语句顺序。
- 创建视图不会冻结其中的数据。源数据及缩放因子须在搬运读取完成前保持有效，不能提前覆盖；原子累加前须初始化目标，且不得与同一区域的未同步普通写入并发。
- 调用本接口不等于 Host 已等待设备执行完成。Host 读取 NPU 结果前须等待设备执行完成，例如调用 `torch.npu.synchronize()`。

## 调用示例

以下示例通过一个 UB 槽位，将 16 个 `float32` 元素从一个 GM Tensor 搬运到另一个 GM Tensor，并在 Host 侧逐元素校验。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
import cannbotdsl as cb
import torch
import torch_npu


@cb.kernel
def copy_kernel(src: cb.Tensor, dst: cb.Tensor):
    slots = cb.RingBuffer(
        cb.MemLoc.UB, shape=(16,), dtype=cb.dtypes.float32, depth=1
    )
    cb.mem_copy(slots.next_write(), src)
    cb.mem_copy(dst, slots.next_read())


@cb.host
def copy(src: cb.Tensor, dst: cb.Tensor):
    copy_kernel[1](src, dst)


def main():
    expected = torch.arange(16, dtype=torch.float32)
    src = expected.to("npu:0")
    dst = torch.empty_like(src)
    copy(src, dst)
    torch.npu.synchronize()
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)
    print("mem_copy example passed")


if __name__ == "__main__":
    main()
```

预期结果：输出 Tensor 的形状为 `(16,)`，元素依次为 `0.0`～`15.0`，与输入逐元素一致；校验通过后打印 `mem_copy example passed`。
