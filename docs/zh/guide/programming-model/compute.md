# 三类计算单元

AI Core 上有三条计算路径，它们的编程模型完全不同。写算子前先想清楚：这段计算该交给谁。

| 路径 | 执行单元 | 流水 | 数据来源 | 入口 |
| --- | --- | --- | --- | --- |
| 矩阵计算 | AIC（Cube） | `PIPE_M` | L0A / L0B，结果在 L0C | [`matmul`](/zh/api/kernel/cube_compute/matmul.html) |
| 矢量计算 | AIV（Vector） | `PIPE_V` | 矢量寄存器，经 UB 中转 | `with vf(mode="simd"):` 内的 `v*` 接口 |
| 标量计算 | Scalar | `PIPE_S` | 标量寄存器、GM | [标量计算](/zh/api/kernel/scalar_compute/)接口 |

矢量这一路还有第二种编程模式：`with vf(mode="simt"):`。它和 SIMD 跑在同一个 AIV 上，但编程模型完全不同，见[本章末尾的 SIMT 模式](/zh/guide/programming-model/compute#simt-模式)。按官方定位，混合编程「以 SIMD 为主、SIMT 为辅」，所以本章主体仍讲 SIMD。

## 矩阵计算

### 计算语义

$$
C_{M \times N} = A_{M \times K} \times B_{K \times N} + C_{M \times N}
$$

这是接口的**数学语义**：`lhs` 是 \(M \times K\)，`rhs` 是 \(K \times N\)。`init=True` 时 C 初值清零；`init=False` 时不清零，初值来源于 L0C Buffer 或 BiasTable Buffer。这就是 K 方向累加的实现方式。

读样例时不要把 GM 上的物理 shape 直接当成上式。仓库 `matmul` 样例和若干官方示例里，右矩阵在 GM 上常按 `(N, K)` 存放，搬进 L0B 时带转置，精度核对用 `A @ B.T`。对官方公式来说，那份 GM `B` 是 \(B_{\text{math}}^{\mathrm{T}}\)。

```python
def matmul(dst, lhs, rhs, *, bias=None, init=True,
           unit_flag=0, disable_gemv=True) -> None
```

| 矩阵 | 存储位置 | 形状 | 数据格式 | 分形大小 |
| --- | --- | --- | --- | --- |
| A | L0A | M×K | Nz | 16×K0 |
| B | L0B | K×N | Zn | K0×16 |
| C | L0C | M×N | Nz | 16×16 |
| Bias | BiasTable | 1×N（广播到 M 行） | ND | — |

`K0 = 32B / sizeof(dtype)`。M、N、K 的单位是元素，取值范围 [0, 4095]；三者任一为 0 时接口视为空操作。

::: warning 这个范围是约定，不是校验
`matmul()` 的 Python 层和底层的 `MmadOp` 验证都**不检查** M/N/K 是否落在 [0, 4095]——CAPI 用 `uint16_t` 承载维度，越界会静默回绕。超了不会报错，只会算错。tiling 阶段自己保证。
:::

常用 dtype 组合：fp16×fp16→fp32、bf16×bf16→fp32、fp32×fp32→fp32、int8×int8→int32，以及 HiFloat8 / FP8 的多种组合，完整表见 [`matmul`](/zh/api/kernel/cube_compute/matmul.html)。

### 四步流程

```text
① GM ──mem_copy(engine=nd2nz)──► L1       搬入 A、B（ND 输入随路转分形）
                                          Bias 和随路量化系数也先进 L1
② L1 ──mem_copy(transpose=?)──► L0A/L0B   按输入布局选择是否转置
   L1 ──mem_copy──► BIAS                  Bias 进专用 Buffer
③ matmul(l0c, l0a, l0b, init=...)         M/K/N 由入参 Tensor 的 shape 决定
                                          首块 init=True，后续 K 块 init=False
④ L0C ──mem_copy(deq_scale=?, relu=?)──► GM / UB / L1
                                          可配置随路量化、激活和输出格式
```

一个完整的 16×16 例子：

```python
@kernel
def _matmul_kernel(a, b, c):
    l1a = Channel(MemLoc.L1, shape=(16, 16), dtype=dtypes.float16, depth=1)
    l1b = Channel(MemLoc.L1, shape=(16, 16), dtype=dtypes.float16, depth=1)
    l0a = Channel(MemLoc.L0A, shape=(16, 16), dtype=dtypes.float16, depth=1)
    l0b = Channel(MemLoc.L0B, shape=(16, 16), dtype=dtypes.float16, depth=1)
    l0c = Channel(MemLoc.L0C, shape=(16, 16), dtype=dtypes.float32, depth=1)

    mem_copy(l1a.produce(), a, engine=make_copy_engine(format_transform="nd2nz"))
    mem_copy(l1b.produce(), b, engine=make_copy_engine(format_transform="nd2nz"))
    mem_copy(l0a.produce(), l1a.consume())
    mem_copy(l0b.produce(), l1b.consume())
    matmul(l0c.produce(), l0a.consume(), l0b.consume(), init=True)
    mem_copy(c, l0c.consume())
```

### K 方向累加

实际矩阵乘的 K 往往远大于 L0A 能放下的量，所以要切 K 并累加：

```python
l0c_acc = l0c.produce()                       # 整个 K 循环共用同一块 L0C
for k_l1_idx in range(k_l1_tiles):
    ...                                        # GM → L1
    for k_l0_idx in range(k_l0_per_l1):
        ...                                    # L1 → L0A/L0B
        global_k = k_l1_idx * k_l0_per_l1 + k_l0_idx
        dsl_matmul(l0c_acc, l0a_tensor, l0b_tensor, init=(global_k == 0))
mem_copy(gm_c_tile, l0c_acc)                   # 累加完成后一次搬出
```

在 NPU ARCH 3510 上，**累加到同一块 L0C 的相邻两次 `matmul` 之间无需额外插入同步指令**。

### 对齐与容量约束

| 项 | 要求 |
| --- | --- |
| `dst`（L0C）起始地址 | 1024 字节对齐 |
| `lhs` / `rhs`（L0A/L0B）起始地址 | 512 字节对齐 |
| `bias`（BiasTable）起始地址 | 64 字节对齐 |
| 申请存储时的补齐 | M、N 向上补齐到 16 的倍数；K 向上补齐到 K0 的倍数 |
| Ascend 950PR/950DT 容量 | L0C 256KB，L0A/L0B 各 64KB，BiasTable 4KB |

### 几个关键特性

| 特性 | 接口 | 说明 |
| --- | --- | --- |
| UnitFlag | `matmul(..., unit_flag=2/3)` | 矩阵乘与搬出的细粒度并行：每算完一个分形就搬出。乘加与搬出指令必须同时开启，开启后无需再插同步 |
| GEMV | `matmul(..., disable_gemv=False)` | M=1 时的向量–矩阵乘专用模式 |
| HF32 | [`enable_hf32`](/zh/api/kernel/cube_compute/enable-hf32.html)、[`set_hf32_round_mode`](/zh/api/kernel/cube_compute/set-hf32-round-mode.html) | fp32 精度换吞吐 |
| FP8 / HiF8 | [`enable_fp8`](/zh/api/kernel/cube_compute/enable-fp8.html)、[`enable_hif8`](/zh/api/kernel/cube_compute/enable-hif8.html) | 低精度矩阵乘 |
| MMAD 方向 | [`set_mmad_direction`](/zh/api/kernel/cube_compute/set-mmad-direction.html) | 开启 UnitFlag 时必须与搬出读取顺序一致 |

开启 UnitFlag 时要注意：同一块 L0C 被多条指令连续操作时，除最后一条外都设 `unit_flag=2`（ENABLE_KEEP）维持占用状态，最后一条设 `3`（ENABLE_UPDATE）解除。搬出开启 NZ2ND 随路转换、或开启 B8/B4 量化并触发 Channel Merge 时用 `cube.set_mmad_direction("n")`，其他场景用 `"m"`。

::: warning
`matmul` 仅在 AIC 上生效。应避免 NaN 输入，否则可能产生执行报错；整数类型仅支持饱和模式。
:::

## 矢量计算：Reg 编程模型

### 为什么是寄存器而不是 UB

Reg 矢量计算接口的输入输出是**矢量数据寄存器（`RawVReg`）和掩码寄存器（`Mask`）**，而不是 UB。

```text
UB ──vload──► 寄存器 ──计算──► 寄存器 ──计算──► 寄存器 ──vstore──► UB
                   └────── 中间结果不落 UB ──────┘
```

传统的「UB → 算 → UB → 算 → UB」模型每一步中间结果都要写回再读出。Reg 模型把整条计算链留在寄存器里，**大幅减少 UB 的反复读写**，这是 Vector 算子性能的主要来源。

代价是你要自己管理搬入搬出和掩码。

::: tip 先读 lane 模型
一个矢量寄存器宽 **256 字节**（VL），fp32 下是 64 条 lane、fp16 下是 128 条；一共 **32 个**数据寄存器，超了会溢出到 UB。掩码宽 VL/8 = 32 字节，每 lane 一位。这几个数字是理解下面所有接口的前提，详见[矢量寄存器与 lane 模型](/zh/guide/programming-model/vector-registers)。
:::

### VF 作用域

所有 Reg 矢量计算接口**必须在 VF 作用域内调用**，不支持在核函数中直接调用：

```python
with vf(mode="simd"):
    mask, _ = update_mask(64, 32)
    acc = vadd(vload(in0, 0), vload(in1, 0), mask=mask)
    vstore(res, 0, acc, mask)
```

VF 作用域的流水类型是 `PIPE_V`。区域内如存在 UB 地址重叠或跨流水依赖，需要按具体接口约束插入 [`vmem_bar`](/zh/api/kernel/reg_compute/reg_sync/vmem-bar.html)。

::: warning 进 VF 之前要做两件事
**一、不要在 VF 区域内调用 `Channel.produce()` / `consume()`。** 在进入 `with vf(...)` 之前选好槽位，把返回的 Tensor 传进去。

**二、不要在 VF 区域内访问对象的成员变量。** 读 `self._xxx` 可能让成员变量从栈溢出到矢量寄存器，既挤占寄存器预算，也会**阻断 VF 融合**。先取成局部变量：

```python
@jit
def _compute(self, x, out):
    w, seg = self._w, self._num_seg      # 先取出来
    with vf(mode="simd"):
        for j in range(seg):             # VF 内只用局部变量
            ...
```
:::

### 掩码：处理非整除长度的标准做法

矢量长度 `VL`（样例中取 64）通常无法整除实际数据长度。设备循环不支持 `break`，所以尾块用**掩码**而不是分支处理：

```python
VL = 64

with vf(mode="simd"):
    full = full_mask()                              # 全开掩码
    remaining = mask_counter(num_col)               # 剩余元素计数器
    for j in range(col_loops):
        preg, remaining = update_mask(remaining, elem_bits=32)  # 每轮消耗一个 VL
        x = vload(x_buf, j * VL)
        y = vmul(x, scale, mask=preg)               # 超出范围的 lane 被屏蔽
        vstore(y_buf, j * VL, y, preg)
```

`update_mask` 既可以从一个计数器迭代推进，也可以直接给元素数量：

```python
preg = update_mask(num_col, elem_bits=32)[0]        # 单次使用
mask, _ = update_mask(64, 32)                       # 位置参数形式
```

::: danger
支持 `mask` 参数的接口，`mask` 必须通过[掩码寄存器操作](/zh/api/kernel/reg_compute/reg_mask/)接口**预先赋值**后再传入。未赋值的掩码寄存器内容不确定，会导致有效元素位置错误。
:::

### 接口分类速查

| 想做什么 | 接口 |
| --- | --- |
| 搬入（连续对齐） | [`vload`](/zh/api/kernel/reg_compute/load/vload.html) |
| 搬入（非连续 / 非对齐 / 广播 / 下采样 / 解包） | `vload_strided`、`vload_unalign`、`vload_broadcast`、`vload_downsample`、`vload_unpack` |
| 搬出（连续对齐 / 打包 / 非对齐 / 掩码） | `vstore`、`vstore_pack`、`vstore_unalign_post`、`vmask_store` |
| 离散聚合 / 散射 | `vgather`、`vgather_datablock`、`vgather_reg`、`vscatter` |
| 基础算术 | `vadd`、`vsub`、`vmul`、`vdiv`、`vexp`、`vlog`、`vsqrt`、`vabs`、`vneg`、`vmax`、`vmin` |
| 标量操作数版本 | 带 `s` 后缀：`vadds`、`vmuls`、`vmaxs`、`vmins` |
| 融合计算 | `vmadd`、`vaxpy`、`vmula`、`vexp_sub`、`vcast_exp_sub`、`vabs_sub`、`vrelu`、`vleakyrelu`、`vprelu` |
| 归约 | `vreduce_sum`、`vreduce_max`、`vreduce_min`、`*_datablock` 变体、`vpair_reduce_sum` |
| 比较 | `veq`、`vne`、`vgt`、`vge`、`vlt`、`vle`（及 `s` 后缀版本），结果是掩码 |
| 选择 | `vselect(a, b, cond_mask=...)` |
| 广播 | `vdup`、`vdups` |
| 类型转换 | `vcast`、`vreinterpret`、`vceil`、`vfloor`、`vtrunc` |
| 排布变换 | `vpack`、`vunpack`、`vinterleave`、`vdeinterleave`、`vmerge`、`vcompress`、`vsqueeze` |
| 掩码操作 | `full_mask`、`create_mask`、`update_mask`、`mask_counter`、`mask_and/or/xor/not` |
| 索引 | `varange` |
| 直方图 | `vhistogram_accumulate`、`vhistogram_frequency` |

完整列表见 [Reg 矢量计算](/zh/api/kernel/reg_compute/)。

### 性能相关的三条规则

**1. Hardware Loop**

满足 Hardware Loop 编码规范的循环会被编译器降成硬件循环，否则退化为软件循环。这是**编译器对生成代码的约束**，不是让你在 Python 里声明 `uint16_t`。写成 `for i in range(n):` 或 `cbd.range(n)` 即可。

要让循环被降成硬件循环，生成结果需要满足：

- 迭代从 0 开始，步长为 1（对应硬件侧的 `uint16_t` 计数器）。
- 循环内没有运行时跳转（`if`/`else`、三元表达式会阻碍 Hardware Loop 生成）。
- 循环计数/边界一旦执行不允许更改。
- 要用外层循环计数作为内层边界时，先把外层计数赋给标量再用。
- **最多四层嵌套**；同一层可以有多个串行循环。
- VF 内的控制流**只支持 `for` 和条件判断**，不支持 `switch`、`while`、`do-while`。

::: danger 退化的代价不止是慢一点
不满足规范时，循环会**静默**退化成软件循环——不报错、不警告。而且退化会连带**阻断下游的 VF 融合**，于是中间结果被迫落 UB，损失可能远大于循环本身的开销。

所以调 VF 性能时，第一件事是确认循环真的满足规范，而不是先去加 `unroll`。
:::

**用循环代替条件判断。** 既然循环体内的 `if` 会阻碍 Hardware Loop，而矢量侧的循环执行又远快于条件跳转，就有一个很实用的改写：把「要不要做」变成「做几次」。

```python
# ✗ 循环体内的运行时分支，阻碍 Hardware Loop
if has_tail:
    process_tail()

# ✓ has_tail 为 0 或 1，循环执行 0 次或 1 次，等价且不引入跳转
for _ in cbd.range(has_tail):
    process_tail()
```

能用 `const_expr` 在编译期消掉的分支优先消掉；消不掉的，优先提到循环外；都做不到的，考虑这个改写。

**2. 指令双发**

处理器可以在同一时钟周期发射两条指令。VF 不是写得越长越好：

- 指令过多会触发 ICache Miss，即使循环间无依赖也无法双发。
- 依赖链过长时用 `cbd.range(..., unroll=N)` 展开循环，提升双发机会。
- 循环内包含同步时，切分循环把同步外提，减少同步次数。
- 适当把中间结果搬出到 UB，减少数据依赖。

**3. VF 融合**

编译器会把控制流等价的相邻 VF 自动融合。你能做的是：

- **把可融合的相邻矢量操作放在同一个 VF 作用域内**，中间结果可以不经过 UB 中转。
- **把需要独立同步或独立流水的操作拆到不同 VF 作用域**，避免不必要的保守同步。

### 一个完整片段

`rms_norm` 里计算 `y = x · rstd · γ` 的核心循环，展示了搬入、解包、转换、计算、打包搬出的完整链路：

```python
@jit
def _compute_y(self, x_ch, gamma_buf, y_buf, rstd_buf, w, cur_rf):
    with vf(mode="simd"):
        full = full_mask()
        col_loops = math.ceil(self._nca / VL)
        for k in cannbotdsl.range(cur_rf):
            x_base = k * w
            rstd_brc = vload_broadcast(rstd_buf, k)          # 标量广播到全 lane
            for j in range(col_loops):
                off = j * VL
                if const_expr(is_16bit):
                    xu = vload_unpack(x_ch, x_base + off,
                                      unpack_mode=UnpackMode.B16_TO_B32)
                    gu = vload_unpack(gamma_buf, off,
                                      unpack_mode=UnpackMode.B16_TO_B32)
                    x = vcast(xu, dtypes.float32, mask=full)
                    g = vcast(gu, dtypes.float32, mask=full)
                else:
                    x = vload(x_ch, x_base + off)
                    g = vload(gamma_buf, off)
                yval = vmul(vmul(x, rstd_brc, mask=full), g, mask=full)
                if const_expr(is_16bit):
                    vstore_pack(y_buf, x_base + off,
                                vcast(yval, out_dtype, mask=full,
                                      rounding=RoundingMode.RN),
                                full, pack_mode=PackMode.B32_TO_B16)
                else:
                    vstore(y_buf, x_base + off, yval, full)
```

注意 16bit 路径的处理：用 `vload_unpack` 把 bf16/fp16 在搬入时就展开成 32 位 lane，中间按 fp32 计算，最后用 `vstore_pack` 在搬出时压回 16 位。整条链路没有额外的 UB 往返。

## SIMT 模式

::: warning CANN 版本要求
使用 SIMT（包括 SIMD + SIMT 混合编程和纯 SIMT 编程）需要 CANN 9.2.0 及以上版本。
:::

::: info 包内可用 · 文档站未收录
本节依据 `cannbotdsl/lang/vf.py` 与 `cannbotdsl/ops/simt.py` 的包源码整理，并与昇腾 950 NPU 架构白皮书的描述对齐。官方 API 文档站目前没有 SIMT 分区，接口稳定性没有承诺。
:::

### 它解决什么问题

SIMD 的 lane 是靠掩码控制的：每条 lane 做同样的事，只有「参不参与」的区别。遇到**不规则控制流**和 **gather / scatter**，掩码就不够用了——你得把分支铺平成「做几次」，把离散访存拆成一堆 `vgather`。

SIMT 换一种抽象：一个指令驱动多个线程，**每个线程独立计算地址，也可以走不同的控制流**。白皮书的分工建议很直接：

> 对于以规则访存为主的 element-wise 计算，优先采用 SIMD 模式以获得高带宽与高算力利用率；而对于不规则或包含分支的部分，采用 SIMT 模式以缓解 gather/scatter 操作带来的控制复杂度。

SIMD 与 SIMT 都以 **VF（Vector Function）** 标出计算区域，一个 VF 只选一种模式。一个 Kernel 可以串接两类 VF，也可以只包含 SIMT VF；这两种 Kernel 的**线程数来源和 Host 启动形式不同**。

### 两种 SIMT 编程模型

| 模型 | 设备侧 VF | Host 侧启动 | 线程数来源 |
| --- | --- | --- | --- |
| SIMD + SIMT 混合 | 同一 Kernel 中既有 SIMD 计算，又有 `vf(mode="simt", thread=N)` | `kernel[block_count](...)` | SIMT VF 的 `thread=N`；Host 不传 thread 维度 |
| 纯 SIMT | 仅有 SIMT VF，使用 `vf(mode="simt")` | `kernel[dim3(block), dim3(thread), dyn_ub_buf](...)` | Host 的 `dim3(thread)`，设备侧通过 `simt.thread_dim()` 获取 |

这里的 `thread=N` 是 **VF 参数**，而 `dim3(thread)` 是 **Host 启动参数**；不能互换。`vf()` 的完整签名是 `vf(*, mode="simd", thread=None, unroll=1, outputs=None)`。两个模式都不接受 `unroll != 1` 或 `outputs` 非默认值；循环展开应写在 `cbd.range()` 上。

### SIMD + SIMT 混合编程

同一 Kernel 先用 SIMD 处理规则数据，再用 SIMT 处理逐线程逻辑。下例中 SIMD 把值写入 UB，SIMT 线程从 UB 读取后写出。

```python
import cannbotdsl as cbd
from cannbotdsl import MemLoc, Tensor, dtypes
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, vdups, vstore


@cbd.kernel
def mixed_kernel(dst: Tensor):
    buf = Buffer(MemLoc.UB, (128,), dtypes.float32)
    with vf(mode="simd"):
        mask = full_mask()
        value = vdups(1.0, dtypes.float32, mask=mask)
        vstore(buf, 0, value, mask)
        vstore(buf, 64, value, mask)
    with vf(mode="simt", thread=128):
        i = cbd.simt.thread_idx()[0]
        if i < cbd.simt.thread_dim()[0]:
            dst[i] = buf[i]


@cbd.host
def run_mixed(dst: Tensor):
    mixed_kernel[1](dst)  # 整数 block 数；线程宽度在 SIMT VF 中指定
```

混合 Kernel 的 Host 启动形式与 SIMD Kernel 一样，只指定 block 数。`thread=128` 只控制当前 SIMT VF 的线程宽度；相邻的 SIMD VF 仍按 SIMD 寄存器与 lane 模型编程。

### 纯 SIMT 编程

如果 Kernel 只有逐线程 SIMT 计算，用 `vf(mode="simt")`，并由 Host 显式给出 block 网格和每个 block 的线程维度。

```python
import cannbotdsl as cbd
from cannbotdsl import Tensor, dim3
from cannbotdsl.lang.vf import vf


@cbd.kernel
def pure_kernel(src: Tensor, dst: Tensor):
    with vf(mode="simt"):
        i = cbd.simt.block_idx()[0] * cbd.simt.thread_dim()[0] + cbd.simt.thread_idx()[0]
        if i < 256:
            dst[i] = src[i]


@cbd.host
def run_pure(src: Tensor, dst: Tensor):
    pure_kernel[dim3(2), dim3(128), 0](src, dst)
```

`dim3(x, y=1, z=1)` 可描述一至三维 block 网格与线程布局。这里启动 2 个 block，每个 block 128 个线程；`simt.block_idx()`、`simt.thread_idx()` 和 `simt.thread_dim()` 用来计算全局位置。`dyn_ub_buf` 是可选的非负整数，省略时为 0，因此也可以写成 `pure_kernel[dim3(2), dim3(128)](src, dst)`。

### Host 启动规则

两种入口都必须由 `@host` 函数体直接启动；普通 Python 调用 `@host`，或用 `cbd.compile(host_fn, ...)` 提前编译，规则与其他 Kernel 相同。区别只在方括号内的配置：

- **混合 SIMD + SIMT：** `kernel[block_count](...)`，`block_count` 是整数；SIMT VF 显式指定 `thread=N`。不能传 Host 线程维度或 `dyn_ub_buf`。
- **纯 SIMT：** `kernel[dim3(block), dim3(thread), dyn_ub_buf](...)`，前两项都必须是 `dim3`；`dyn_ub_buf` 可省略。每个 block 的线程数为 `thread.x * thread.y * thread.z`，范围是 1 到 1024。

纯 SIMT 不能在设备侧写固定的 `vf(mode="simt", thread=N)` 后再从 Host 传 `dim3(thread)`；混合 Kernel 也不能用纯 SIMT 的 `dim3` 启动形式。编译器会校验这两类配置。

### 接口分类

`cannbotdsl.ops.simt`（顶层可用 `cbd.simt`）按线程模型的通用命名组织：

| 类别 | 接口 |
| --- | --- |
| 线程与网格 | `thread_idx`、`thread_dim`、`block_idx`、`grid_dim`、`warp_size`、`lane_idx`、各类 lanemask |
| 标量数学 | `cast`、`bitcast`、`sqrt`、`rsqrt`、`exp`、`log`、`abs`、三角函数、`fma` |
| 短向量 | `make_short_vector`、`v*` 运算、`vcast`、`vbitcast`、`vpack_*` |
| 同步与 Cache | `syncthreads`、`threadfence*`、`l2cache_load` / `store` / `vload` / `vstore`、`dcci_*`、`nop` |
| Warp 级 | `all`、`any`、`ballot`、`activemask`、`shfl*`、`reduce_*` |
| 原子 | `atomic_*`、`vatomic_*` |


### 两个代价

**一、SIMT 会从 UB 划走一块做 Data Cache。** 混用两种模式时，UB 预算按最紧的一侧算。实践中常见的口径是 SIMD 下按 248 KB、开 SIMT 后按 216 KB 算（即再让出 32 KB），但这个数不在 `get_mem_size` 的返回值里——**要自己留**。

**二、线程数越多，每线程寄存器越少。** 这是 SIMT 寄存器文件（总容量固定）的分配规则，档位大致是：线程数翻倍、每线程寄存器减半。计算密集的算子在默认的大线程数下容易**寄存器溢出到栈**（栈在 Global Memory 上），性能反而掉下来。症状和调法与 SIMD 侧的寄存器溢出同理，见[矢量寄存器与 lane 模型](/zh/guide/programming-model/vector-registers#寄存器预算)——区别是 SIMT 这一侧的旋钮是**线程数**，SIMD 那一侧是 **VF 长度**。

### 什么时候真的该用它

| 情况 | 选哪个 |
| --- | --- |
| 规则的 element-wise、归一化、归约 | **SIMD**。这是主路径，不要无谓地改用 SIMT |
| 控制流复杂、想先跑通再优化 | 先用 SIMT 过渡，热点再改 SIMD |
| 每个元素走的分支不一样（不规则控制流） | SIMT |
| 重度 gather / scatter，`vgather` 写起来太绕 | SIMT |
| 需要 warp 级原语（`shfl`、`ballot`） | SIMT |

::: warning 别把它当成 SIMD 的替代品
白皮书的原话是「以 SIMD 为主、SIMT 为辅」。SIMD 这一侧有双发 ALU 和乱序执行，单位周期吞吐更高；SIMT 换来的是编程便利，不是性能。先用 SIMD 写，撞到控制流复杂度墙了再考虑把**那一段** VF 换成 SIMT。
:::

## 标量计算

标量流水（`PIPE_S`）用于地址计算、循环控制和轻量数据处理：

| 分类 | 接口 |
| --- | --- |
| 位运算 | `popc`、`clz`、`ffs`、`ffz`、`set_nthbit`、`clear_nthbit`、`sflbits`、`zero_bits_cnt` |
| 类型转换 | `cast`（配合 `RoundingMode`） |
| 不经 DCache 的 GM 访存 | `load_bypass`、`vec_store_bypass`、`cube_store_bypass` |

::: warning
Tensor 下标读写（`y[i] = x[j]`）走的是标量流水，**只适合搬少量标量**。批量数据用标量循环搬运会比矢量路径慢几个数量级。
:::

## 原子操作

[原子操作](/zh/api/kernel/atomic/)在 GM 地址上做单点原子计算，多个 AI Core 的操作串行化执行（绕过 DCache）：

`atomic_add`、`atomic_sub`、`atomic_max`、`atomic_min`、`atomic_and`、`atomic_or`、`atomic_xor`、`atomic_cas`、`atomic_exch`、`atomic_inc`、`atomic_dec`。

批量累加不要用标量原子接口，而是用 `mem_copy(..., atomic_add=True)` 在搬运时完成——支持 int8/int16/int32/bf16/fp16/fp32，仅支持 UB/L0C → GM。

## 选哪条路径

| 计算特征 | 路径 |
| --- | --- |
| 两个矩阵相乘、卷积（implicit GEMM） | Cube |
| 逐元素运算、归一化、激活、softmax、归约 | Vector / SIMD（`vf(mode="simd")`） |
| 不规则控制流、重度 gather / scatter、需要 warp 原语 | Vector / SIMT（`vf(mode="simt")`） |
| 地址计算、tile 编号、分支判断、少量元数据 | Scalar |
| 多核把结果累加到同一块 GM | `mem_copy(..., atomic_add=True)` 或原子接口 |
| 矩阵乘之后紧接逐元素后处理 | Mix 算子：AIC 做 Cube，AIV 做 Vector，用 `ChannelKind.CrossCore` 交接 |

## 下一步

[数据类型与量化](/zh/guide/programming-model/data-types)：三条路径都支持低精度，scale 存在哪里、谁负责搬。
