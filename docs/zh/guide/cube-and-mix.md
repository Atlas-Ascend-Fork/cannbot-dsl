# 第一个 Cube 算子与第一个 Mix 算子

[写出第一个算子](./first-operator.md)走完了纯 Vector 的路线。但仓库里的旗舰样例 `matmul` 和 `flash_attn` 都在另外两条路上：**Cube**（只有矩阵乘）和 **Mix**（AIC 与 AIV 协作）。这一章补上这两级台阶，每一步只加一个新概念。

## Step 1：16×16 的最小矩阵乘

先把 Cube 的四步流程跑通，shape 写死成一个分形的大小。

```python
import torch
import torch_npu  # noqa: F401

from cannbotdsl import RingBuffer, MemLoc, dtypes, host, make_layout
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.ops.matmul import matmul
from cannbotdsl.ops.memcpy import mem_copy


@kernel
def _matmul_kernel(a, b, c):
    l1a = RingBuffer(MemLoc.L1, dtypes.float16, make_layout((16, 16)), depth=1)
    l1b = RingBuffer(MemLoc.L1, dtypes.float16, make_layout((16, 16)), depth=1)
    l0a = RingBuffer(MemLoc.L0A, dtypes.float16, make_layout((16, 16)), depth=1)
    l0b = RingBuffer(MemLoc.L0B, dtypes.float16, make_layout((16, 16)), depth=1)
    l0c = RingBuffer(MemLoc.L0C, dtypes.float32, make_layout((16, 16)), depth=1)

    # ① GM → L1，目的布局决定随路 ND → NZ 转换
    mem_copy(l1a.next_write(), a)
    mem_copy(l1b.next_write(), b)

    # ② L1 → L0A / L0B，B 的轴序用 permute 表达
    mem_copy(l0a.next_write(), l1a.next_read())
    mem_copy(l0b.next_write(), l1b.next_read().permute(1, 0))

    # ③ 乘加，返回实际目的 View
    acc = matmul(
        l0c.next_write(), l0a.next_read(), l0b.next_read(), init=True
    )

    # ④ L0C → GM
    mem_copy(c, acc)


@host
def run(a, b, c):
    _matmul_kernel[1](a, b, c)
```

和 Vector 算子相比，三个新东西：

- **存储层级多了一层。** 矩阵乘的输入**必须经 L1 中转**，不能从 GM 直接进 L0A / L0B。
- **格式和轴序由 Tensor 表达。** L1 / L0 的默认 `data_format` 是 `"nz"`，GM 上是 ND，所以 GM→L1 直接 `mem_copy` 就会按源、目标布局执行随路转换；B 的转置用 `permute`，不再传 `engine` 或 `transpose=True`。
- **没有 `with vf(...)`。** Cube 的接口直接作用在 Tensor 上，不经寄存器。

::: tip 为什么是 16×16
Cube 一拍完成一个 fp16 的 16×16 × 16×16 矩阵乘，这是硬件的最小计算单位。所有 tile 尺寸最终都要对齐到它：**M、N 对齐 16，K 对齐 K0 = 32 字节 / 元素字节数**（fp16 下 K0 = 16）。
:::

## Step 2：切 K 并累加

真实矩阵乘的 K 远大于 L0A 能装下的量，所以要切 K，并且**在同一块 L0C 上累加**。

```python
l0c_acc = l0c.next_write()                   # 整个 K 循环共用一块 L0C
for k_idx in range(k_tiles):
    gm_a_tile = tile_slice(gm_a, (M, K_TILE), (0, k_idx))
    gm_b_tile = tile_slice(gm_b, (K_TILE, N), (k_idx, 0))
    mem_copy(l1a.next_write(), gm_a_tile)
    mem_copy(l1b.next_write(), gm_b_tile)
    mem_copy(l0a.next_write(), l1a.next_read())
    mem_copy(l0b.next_write(), l1b.next_read().permute(1, 0))
    l0c_acc = matmul(
        l0c_acc, l0a.next_read(), l0b.next_read(), init=(k_idx == 0)
    )

mem_copy(gm_c, l0c_acc)                      # 累加完一次搬出
```

关键只有一个参数：**`init=True` 清零 L0C，`init=False` 在原值上累加。** 第一块用 `True`，后续全部用 `False`。`matmul` 返回实际目的 View，把它继续传给下一次累加和最终搬出，可以保留动态尾块的布局元数据。

两个免费的便利：

- **累加到同一块 L0C 的相邻两次 `matmul` 之间不需要插同步**（NPU ARCH 3510 的特性）。
- `init=False` 时 L0C 的初值也可以来自 BiasTable，这就是 bias 被折进乘加的方式。

::: warning 别把 `init` 写成按 RingBuffer 轮转
`l0c_acc` 必须在 K 循环**之外** `next_write()` 一次。如果在循环里每轮都选新槽，累加关系就会断掉。
:::

## Step 3：多核 + 两级切块

现在加上分核和 L1→L0 的第二级切块。这一步的结构就是 `samples/matmul` 的骨架了。

```python
@kernel
class MatmulKernel:
    def __init__(self, t):                  # t 是 Host 侧算好的 tiling 参数
        self.t = t
        self._l1_a = RingBuffer(
            MemLoc.L1, t.a_dtype, make_layout((t.base_m, t.k_l1)),
            depth=t.l1_buffer_num,
        )
        self._l1_b = RingBuffer(
            MemLoc.L1, t.b_dtype, make_layout((t.base_n, t.k_l1)),
            depth=t.l1_buffer_num,
        )
        self._l0a = RingBuffer(
            MemLoc.L0A, t.a_dtype, make_layout((t.base_m, t.base_k)), depth=2
        )
        self._l0b = RingBuffer(
            MemLoc.L0B, t.b_dtype, make_layout((t.base_k, t.base_n)), depth=2
        )
        self._l0c = RingBuffer(
            MemLoc.L0C, dtypes.float32, make_layout((t.base_m, t.base_n)),
            depth=t.l0c_db,
        )

    def __call__(self, gm_a, gm_b, gm_c):
        t = self.t
        bi, bn = get_block_idx(), get_block_num()
        n_tiles = ceil_div(t.n, t.base_n)

        for tile_idx in cannbotdsl.range(bi, t.m_tiles * n_tiles, bn):
            m_idx = tile_idx // n_tiles
            n_idx = tile_idx % n_tiles
            l0c_acc = self._l0c.next_write()

            for k_l1_idx in cannbotdsl.range(t.k_l1_tiles):
                # 第一级：GM → L1，格式由目的布局决定
                mem_copy(
                    self._l1_a.next_write(),
                    tile_slice(gm_a, (t.base_m, t.k_l1), (m_idx, k_l1_idx)),
                )
                mem_copy(
                    self._l1_b.next_write(),
                    tile_slice(gm_b, (t.base_n, t.k_l1), (n_idx, k_l1_idx)),
                )
                l1_a_t = self._l1_a.next_read()
                l1_b_t = self._l1_b.next_read()

                # 第二级：L1 → L0
                for k_l0_idx in range(t.k_l0_per_l1):
                    mem_copy(
                        self._l0a.next_write(),
                        tile_slice(l1_a_t, (t.base_m, t.base_k), (0, k_l0_idx)),
                    )
                    mem_copy(
                        self._l0b.next_write(),
                        tile_slice(
                            l1_b_t, (t.base_n, t.base_k), (0, k_l0_idx)
                        ).permute(1, 0),
                    )
                    global_k = k_l1_idx * t.k_l0_per_l1 + k_l0_idx
                    l0c_acc = matmul(
                        l0c_acc,
                        self._l0a.next_read(),
                        self._l0b.next_read(),
                        init=(global_k == 0),
                    )

            mem_copy(
                tile_slice(gm_c, (t.base_m, t.base_n), (m_idx, n_idx)),
                l0c_acc,
            )
```

新增的三个要点：

1. **两级 `tile_slice`。** 第一级从 GM 取 L1 块，第二级从 **L1 块**里再取 L0 块。注意第二级的 `coord=(0, k_l0_idx)`——第 0 维取 0 表示覆盖整个 M 轴，第 1 维按 K 滚动。
2. **用 `permute` 表达 B 的轴序。** 样例里 GM 上的右矩阵常按 `(N, K)` 存放，搬进 L0B 时通过视图转置。这时精度核对要用 `A @ B.T`。读代码时先分清「接口的数学语义」和「这份 GM 怎么存」。
3. **`depth` 按预算选。** L1 的 `depth` 在 2 和 4 之间按容量选，L0C 的 `depth` 看 `base_m × base_n × 4 × 2` 是否放得下。tiling 的完整推导见[高性能算子编写指南](./performance.md)。

对齐约束必须满足，否则结果错位：

| 项 | 要求 |
| --- | --- |
| L0C 起始地址 | 1024 字节对齐 |
| L0A / L0B 起始地址 | 512 字节对齐 |
| BiasTable 起始地址 | 64 字节对齐 |
| 申请存储时 | M、N 补齐到 16 的倍数，K 补齐到 K0 的倍数 |

## Step 4：第一个 Mix 算子

Mix 算子的意思是**一个 group 里同时有 AIC（主核）和 AIV（从核）**，两者分工协作。典型场景是矩阵乘之后紧接逐元素后处理：Cube 算 \(A \times B\)，Vector 做 softmax、量化或激活。

### block 与 subblock

| 算子类型 | 执行核 | block / subblock |
| --- | --- | --- |
| Cube 算子 | 仅 AIC | AIC 为 block，无 subblock |
| Vector 算子 | 仅 AIV | AIV 为 block，无 subblock |
| **Mix 算子** | AIC + AIV | AIC 为 block，AIV 为 subblock，比例 1：1 或 1：2 |

**只有 Mix 算子才有 subblock。** 1：2 配置下一个 AIC 配两个 AIV，样例里常这样算出逻辑 AIV 编号：

```python
aiv = get_block_idx() * 2 + get_subblock_id()
```

### 用 RingBuffer 做跨核交接

交接**不要手写核间同步加 GM 中转**。使用普通 `RingBuffer`，编译器会根据实际 Tensor 访问推导跨核属性：

```python
@kernel
class MixKernel:
    def __init__(self, t):
        # AIC 算完的结果放这里，AIV 从这里取
        self._p = RingBuffer(
            MemLoc.L1, dtypes.float16, make_layout((t.base_m, t.base_n)), depth=2
        )
        self._ub_out = RingBuffer(
            MemLoc.UB, dtypes.float16, make_layout((t.base_m, t.base_n)), depth=2
        )

    def __call__(self, gm_a, gm_b, gm_out):
        self._cube_side(gm_a, gm_b)        # AIC：矩阵乘并写入交接槽
        self._vector_side(gm_out)          # AIV：读取交接槽并做后处理
```

::: danger 不要用 `is_aic()` / `is_aiv()` 去分 Mix 的两侧
看到「区分 AIC 与 AIV」很容易想写 `if is_aic(): ... else: ...`。这在 Mix kernel 里是错的：

**在 MIX 编译下，`is_aic()` 和 `is_aiv()` 都返回 `false`。** 它们是**编译期常量谓词**，只在「这份代码被克隆成 AIC 侧 / AIV 侧」时各自为真。MIX 构建是单份代码，两个都是 false，于是两个分支都不会进——而且不报错，只是什么都不做。

它们的正确用途是：在**纯 AIC 或纯 AIV** 的 kernel 里按核类型做编译期特化，不是在 Mix kernel 里分派。
:::

正确的组织方式是**把 Cube 侧和 Vector 侧写成两个类（或两组 `@jit` 方法）**，由 kernel 入口按阶段调用，编译器按核类型裁剪。`samples/flash_attn/flash_attn.py` 就是这个结构（`Cube` 与 `Vector` 两个类），比在一个函数体里写 if-else 清晰得多，也避开了上面那个坑。

::: info 待核实
Mix kernel 的入口分派细节（谁负责把两侧代码路由到对应的核）官方文档站没有专门说明。以 `samples/flash_attn/flash_attn.py` 为模板。
:::

要点：

- 跨核属性由编译器按实际 Tensor 访问推导，不需要 `kind`，`depth` 也没有跨核专用的 8 槽上限。
- 同步仍由编译器按实际访问生成，和同核一样。
- 交接点放在 L1 还是 UB，取决于下游要怎么用。L0C 直接给 AIV 也是可以的（`L0C → UB` 是 3510 新增的通路）。

### 为什么值得拆成 Mix

把矩阵乘和后处理放在同一个核上串行做，Cube 算的时候 Vector 闲着，反之亦然。拆成 Mix 之后两者**流水重叠**：AIC 算第 n 块的同时，AIV 在处理第 n−1 块。这是 attention 类算子性能的主要来源之一。

代价是你要自己想清楚交接的粒度：交接太细，同步开销占比高；交接太粗，片上放不下，而且流水填不满。

## 检查清单

| 项 | 检查 |
| --- | --- |
| 通路 | 矩阵乘输入是否经 L1 中转；GM→L1 目的布局是否正确表达 NZ |
| 累加 | 第一块 `init=True`，后续 `init=False`；`l0c_acc` 是否在 K 循环外取并持续传递返回 View |
| 转置 | GM 上 B 是 `(K, N)` 还是 `(N, K)`；是否用 `permute` 表达轴序，golden 该不该 `A @ B.T` |
| 对齐 | M / N 对齐 16，K 对齐 K0；L0C / L0A / L0B 地址对齐 |
| 容量 | tile 是否放得下 L0A / L0B / L0C / L1，用 `get_mem_size()` 查 |
| 分核 | `block_dim` 是否收敛到实际需要的核数 |
| Mix | 是否真的需要 subblock；交接是否用普通 `RingBuffer` 而非手写同步 |
| 量化 | 低精度路径是否开了对应的 `enable_*`；MX scale 是否单独搬了，`matmul` 是否显式 `mx_mode=True` |

## 下一步

[AI CPU 与调度计划](./aicpu.md)：变长和稀疏场景下，「每个核该干多少活」编译期算不出来，怎么办。

或者按需要跳：

- 想把 attention 这类多阶段算子拆开：[融合算子的设计方法论](./fusion-design.md)
- 想看 tiling 参数怎么搜出来：[高性能算子编写指南](./performance.md)
- 想用低精度：[数据类型与量化](./data-types.md)
