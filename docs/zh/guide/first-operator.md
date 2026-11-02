# 写出第一个算子

这一节从最小的向量加开始，分五步加到「多核 + tiling + double buffer + torch 接口 + 精度测试」，把前面几节的概念串起来。

::: warning 关于本节代码的验证状态
Step 1 改写自 [Reg 矢量计算概述](../api/compute/vector/overview.md)的官方示例。文档构建检查用于验证页面生成和站内链接，**不代表 API 示例已通过设备运行验证**。所以请把它当作**结构正确的起点**，而不是保证可运行的代码——第一次跑的时候对照报错逐步调整是正常的。

Step 2 之后的代码按 `samples/rms_norm` 的结构逐步扩展，用于讲解模式；落到具体算子时请以 `samples/` 下对应样例为准，那些代码有配套测试。
:::

## 准备

```bash
python -m pip install cannbot-dsl
python -c 'import cannbotdsl; print(cannbotdsl.__version__)'
```

需要一台支持的 NPU（Ascend 950PR / Ascend 950DT），并已 `source ${install_path}/ascend-toolkit/set_env.sh`。

## Step 1：能跑起来的最小算子

把下面的代码存成 `reg_add.py`，`python reg_add.py` 运行。

```python
import torch
import torch_npu  # noqa: F401  # 向 PyTorch 注册 Ascend NPU 后端

from cannbotdsl import RingBuffer, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import update_mask, vadd, vload, vstore
from cannbotdsl.tensor import MemLoc


@kernel
def _reg_add_kernel(src0, src1, dst):
    buf0 = RingBuffer(MemLoc.UB, src0.dtype, make_layout((64,)), depth=1)
    buf1 = RingBuffer(MemLoc.UB, src1.dtype, make_layout((64,)), depth=1)
    out = RingBuffer(MemLoc.UB, dst.dtype, make_layout((64,)), depth=1)

    # GM 数据搬运至 UB
    mem_copy(buf0.next_write(), src0)
    mem_copy(buf1.next_write(), src1)

    in0 = buf0.next_read()
    in1 = buf1.next_read()
    res = out.next_write()
    # Vector Function：更新 mask、搬入、计算、搬出
    with vf(mode="simd"):
        mask, _ = update_mask(64, 32)
        acc = vadd(vload(in0, 0), vload(in1, 0), mask=mask)
        vstore(res, 0, acc, mask)

    # UB 数据搬运至 GM
    mem_copy(dst, out.next_read())


@host
def run(src0, src1, dst):
    _reg_add_kernel[1](src0, src1, dst)


def main():
    src0 = torch.arange(64, dtype=torch.float32, device="npu:0")
    src1 = torch.arange(64, dtype=torch.float32, device="npu:0") + 1.0
    dst = torch.empty_like(src0)

    run(src0, src1, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), (src0 + src1).cpu())
    print("reg compute example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")


if __name__ == "__main__":
    main()
```

预期输出：

```text
reg compute example passed
first=1.0000, last=127.0000
```

### 逐行解读

```python
@kernel
def _reg_add_kernel(src0, src1, dst):
```
设备函数。三个参数都是 GM 上的 Tensor，`dst` 是输出——**Kernel 不返回值，输出通过参数传入**。

```python
    buf0 = RingBuffer(MemLoc.UB, src0.dtype, make_layout((64,)), depth=1)
```
在 UB 上申请 1 个长度为 64 的槽位。`make_layout` 描述槽内 Tensor 的逻辑布局，`depth=1` 表示不轮转。地址由框架规划。

```python
    mem_copy(buf0.next_write(), src0)
```
`next_write()` 选择写游标当前槽位并推进写游标，`mem_copy` 把 GM 数据搬进去。**数据同步由框架根据实际 Tensor 访问生成**，这里不需要写 wait。

```python
    in0 = buf0.next_read()
    res = out.next_write()
    with vf(mode="simd"):
```
先在 VF 外把所有要用到的槽位取出来，再进入 VF 作用域——**不要在 VF 内调用 `next_write()` / `next_read()`**。

```python
        mask, _ = update_mask(64, 32)
```
生成一个覆盖 64 个 32 位元素的掩码。`update_mask` 返回 `(mask, remaining)`，这里数据正好一拍算完，剩余计数用不到。

```python
        acc = vadd(vload(in0, 0), vload(in1, 0), mask=mask)
        vstore(res, 0, acc, mask)
```
`vload(buf, offset)` 从 UB 搬到矢量寄存器，`vadd` 在寄存器上算，`vstore` 搬回 UB。偏移单位是元素。

```python
@host
def run(src0, src1, dst):
    _reg_add_kernel[1](src0, src1, dst)
```
Host 入口。`[1]` 是 block 数量。**Kernel 启动语句必须直接写在 `@host` 函数体里。**

## Step 2：处理任意长度 + 多核

Step 1 写死了 64 个元素、1 个核。实际算子要能处理任意长度并用满所有核。

三个变化：

1. 把长度和 tile 大小作为**编译期配置**传给 `@kernel` 类的 `__init__`。
2. 用 `get_block_idx()` / `get_block_num()` 做多核切分。
3. 用 `tile_slice` 按 tile 序号取 GM 视图，并对尾块显式绑定实际范围。

```python
import math

import torch
import torch_npu  # noqa: F401

import cannbotdsl
from cannbotdsl import RingBuffer, dtypes, get_mem_size, make_layout
from cannbotdsl.lang.host import host
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.arch import get_block_idx, get_block_num
from cannbotdsl.ops.memcpy import mem_copy
from cannbotdsl.ops.reg import mask_counter, update_mask, vadd, vload, vstore
from cannbotdsl.tensor import MemLoc, Tensor, tile_slice

VL = 64                       # 矢量长度


def choose_tile(dtype, ub_size):
    elem_bytes = {dtypes.float16: 2, dtypes.bfloat16: 2, dtypes.float32: 4}[dtype]
    # 3 个缓冲（x、y、out）× double buffer 2 级，留 1KB 余量
    budget = (ub_size - 1024) // (3 * 2 * elem_bytes)
    return max(VL, (budget // VL) * VL)


@kernel
class VecAddKernel:
    def __init__(self, n, dtype=dtypes.float32):
        # ── 编译期：算 tile 大小和循环次数 ──
        self._n = int(n)
        self._dtype = dtype
        self._tile = choose_tile(dtype, get_mem_size("ub"))
        self._n_tiles = math.ceil(self._n / self._tile)
        self._seg = self._tile // VL          # 每个 tile 内的 VF 迭代次数

        # ── 声明片上资源 ──
        self._x_rb = RingBuffer(
            MemLoc.UB, dtype, make_layout((self._tile,)), depth=2
        )
        self._y_rb = RingBuffer(
            MemLoc.UB, dtype, make_layout((self._tile,)), depth=2
        )
        self._o_rb = RingBuffer(
            MemLoc.UB, dtype, make_layout((self._tile,)), depth=2
        )

    def __call__(self, gm_x: Tensor, gm_y: Tensor, gm_out: Tensor):
        tile = self._tile
        n_tiles = self._n_tiles
        bi = get_block_idx()
        bn = get_block_num()

        for t in cannbotdsl.range(bi, n_tiles, bn):
            # tile_slice 会裁剪尾块；选中槽也要显式绑定相同的实际范围。
            actual = min(tile, self._n - t * tile)
            mem_copy(
                self._x_rb.next_write()[:actual,],
                tile_slice(gm_x, (tile,), (t,)),
            )
            mem_copy(
                self._y_rb.next_write()[:actual,],
                tile_slice(gm_y, (tile,), (t,)),
            )

            x = self._x_rb.next_read()
            y = self._y_rb.next_read()
            o = self._o_rb.next_write()
            self._add(x, y, o, actual)

            mem_copy(
                tile_slice(gm_out, (tile,), (t,)),
                self._o_rb.next_read()[:actual,],
            )

    @jit
    def _add(self, x, y, o, actual):
        with vf(mode="simd"):
            remaining = mask_counter(actual)
            for s in range(self._seg):
                m, remaining = update_mask(remaining, elem_bits=32)
                acc = vadd(vload(x, s * VL), vload(y, s * VL), mask=m)
                vstore(o, s * VL, acc, m)


class VecAdd:
    @host
    def run(self, gm_x, gm_y, gm_out, block_dim):
        VecAddKernel(gm_x.shape[0])[block_dim](gm_x, gm_y, gm_out)
```

几个要点：

- **`__init__` 里全是普通 Python 算术**。它们在构图期执行，结果固化进设备代码。UB 容量用 `get_mem_size("ub")` 查询，不要写死 `248 * 1024`。
- `for t in cannbotdsl.range(bi, n_tiles, bn)` 是轮转分核：第 `bi` 个核处理第 `bi`、`bi+bn`、`bi+2bn`… 个 tile。
- **尾块搬运和计算都按实际范围处理。** `tile_slice` 返回裁剪后的 GM View；`next_write()[:actual,]` 给目的槽绑定相同范围。`mask_counter(actual)` 建计数器，`update_mask` 每轮消耗一个 VL，超出的 lane 被屏蔽。读端用同一 `actual` 构造输出 View，不依赖选槽恢复写入布局。
- 计算逻辑拆进 `@jit` 方法，`__call__` 只负责调度。这是样例里统一的组织方式。

## Step 3：加上 double buffer

Step 2 的 `RingBuffer` 已经是 `depth=2`，但程序顺序还是「搬完就用」，搬运和计算没有重叠。把下一拍的搬运提前发出去就能重叠：

```python
    def __call__(self, gm_x: Tensor, gm_y: Tensor, gm_out: Tensor):
        tile = self._tile
        n_tiles = self._n_tiles
        bi = get_block_idx()
        bn = get_block_num()

        # ── 预取第一拍 ──
        if bi < n_tiles:
            actual = min(tile, self._n - bi * tile)
            mem_copy(
                self._x_rb.next_write()[:actual,],
                tile_slice(gm_x, (tile,), (bi,)),
            )
            mem_copy(
                self._y_rb.next_write()[:actual,],
                tile_slice(gm_y, (tile,), (bi,)),
            )

        for t in cannbotdsl.range(bi, n_tiles, bn):
            # ── 先发下一拍的搬运 ──
            nxt = t + bn
            if nxt < n_tiles:
                next_actual = min(tile, self._n - nxt * tile)
                mem_copy(
                    self._x_rb.next_write()[:next_actual,],
                    tile_slice(gm_x, (tile,), (nxt,)),
                )
                mem_copy(
                    self._y_rb.next_write()[:next_actual,],
                    tile_slice(gm_y, (tile,), (nxt,)),
                )

            # ── 再使用当前拍 ──
            actual = min(tile, self._n - t * tile)
            x = self._x_rb.next_read()
            y = self._y_rb.next_read()
            o = self._o_rb.next_write()
            self._add(x, y, o, actual)
            mem_copy(
                tile_slice(gm_out, (tile,), (t,)),
                self._o_rb.next_read()[:actual,],
            )
```

这个「**循环外预取一拍，循环内先发下一拍再使用当前拍**」的骨架，几乎所有访存受限的算子都用得上。

一次搬运从发到完成跨过多轮计算时，可以增加 `depth`，让更多块同时在途，而不是只预装几块就停。这时候还需要用 [`Array`](./onchip-memory.md#array保存软件流水的标量历史) 保存每一拍对应的 tile 坐标和其他标量元数据；旧的 `DelayLineGroup` 已删除。

## Step 4：Host 侧包装成 torch 算子

对外接口做三件事：校验、分配输出、决定 `block_dim`。

```python
DEFAULT_BLOCK_NUM = 64

_TORCH_TO_DSL = {
    torch.float16: dtypes.float16,
    torch.bfloat16: dtypes.bfloat16,
    torch.float32: dtypes.float32,
}


def vec_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")
    if x.dtype != y.dtype:
        raise TypeError(f"dtype mismatch: {x.dtype} vs {y.dtype}")
    if x.dtype not in _TORCH_TO_DSL:
        raise TypeError("only float16 / bfloat16 / float32 are supported")
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("inputs must be contiguous")

    out = torch.empty_like(x)
    n = x.numel()
    dsl_dtype = _TORCH_TO_DSL[x.dtype]
    tile = choose_tile(dsl_dtype, get_mem_size("ub"))
    n_tiles = math.ceil(n / tile)

    # 设备侧通过实际 View + 掩码处理尾块，无需仅为搬运而 padding。
    x1d = x.reshape(n)
    y1d = y.reshape(n)
    o1d = out.reshape(n)

    # 按真实 tile 数启动核，不要用 VL 去猜
    block_dim = min(DEFAULT_BLOCK_NUM, max(1, n_tiles))

    VecAdd().run(x1d, y1d, o1d, block_dim)
    return out
```

这一层是**普通 Python**，不受 DSL 约束，可以随意写校验和 reshape。`samples/rms_norm/rms_norm.py` 的 `rms_norm()` 函数是一个更完整的参考（处理任意 rank、推导归一化维度、准备副输出）。

这一层的完整讲法——stream 语义、多输出与原地更新、workspace 分配、注册成 torch 自定义算子——在[torch 接口与 stream 语义](./torch-interop.md)。本节先按能跑的最小版本写。

## Step 5：精度测试

和仓库的测试风格保持一致，用 pytest + torch golden：

```python
# test_vec_add.py
import pytest
import torch
import torch_npu  # noqa: F401

import cannbotdsl
from vec_add import vec_add


@pytest.mark.parametrize("n", [64, 1000, 65536, 1048577])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_vec_add(n, dtype):
    cannbotdsl.clear_compile_cache()

    x = torch.randn(n, dtype=dtype).npu()
    y = torch.randn(n, dtype=dtype).npu()

    out = vec_add(x, y)
    torch.npu.synchronize()

    golden = (x.float() + y.float()).to(dtype)
    torch.testing.assert_close(out.cpu(), golden.cpu(), rtol=1e-3, atol=1e-3)
```

```bash
python3 -m pytest test_vec_add.py -v
```

用例要覆盖：正好整除的长度、带尾块的长度、只有一个 tile 的长度（小于核数）、以及超大长度。这四类是最容易出问题的边界。

## Step 6：看看快不快

向量加是**纯访存受限**算子，它的性能上限就是 HBM 带宽。判断是否达标的方法：

```python
import time

x = torch.randn(1 << 26, dtype=torch.float16).npu()
y = torch.randn_like(x)

vec_add(x, y)               # 预热，触发编译
torch.npu.synchronize()

t0 = time.perf_counter()
for _ in range(100):
    vec_add(x, y)
torch.npu.synchronize()
dt = (time.perf_counter() - t0) / 100

bytes_moved = x.numel() * x.element_size() * 3    # 读 x、读 y、写 out
print(f"{dt * 1e6:.1f} us, {bytes_moved / dt / 1e9:.1f} GB/s")
```

如果带宽明显低于硬件峰值，按[高性能算子编写指南](./performance.md)的顺序排查：先看分核是否均衡，再看 tile 是否太小（搬运次数太多），最后看流水是否真的重叠了。

精细的性能分析用 msprof 采集 Task Duration，见[调试与精度验证](./debugging.md)。

## 一个完整算子的检查清单

| 项 | 检查 |
| --- | --- |
| 结构 | 分成「torch 接口 / Host tiling + `@host` / `@kernel` 设备代码」三层 |
| 编译期配置 | tile 大小、分支选择、循环次数都在 `__init__` 里算好 |
| 片上预算 | UB / L1 / L0 容量用 `get_mem_size()` 查询，不硬编码 |
| 分核 | 用 `get_block_idx()` / `get_block_num()`，tile 数少于核数时不启动多余的核 |
| 尾块 | 用掩码处理计算；搬运前给目的 Tensor 显式绑定实际范围，读端保存或重建相同 View |
| 流水 | RingBuffer `depth >= 2`，循环外预取一拍；深度流水的标量历史用 `Array` 保存 |
| 对齐 | 矢量侧对齐到 VL，Cube 侧对齐到 16 / K0 |
| 校验 | shape / dtype / contiguous 在 torch 接口层检查并抛出清晰异常 |
| 测试 | 覆盖整除、尾块、单 tile、超大四类长度 |
| 文档 | 目录下放一个 README，写清接口签名、数据类型约束和运行方式 |

## 下一步

这一节走的是纯 Vector 路线。矩阵乘和 attention 在另外两条路上，见[第一个 Cube 算子与第一个 Mix 算子](./cube-and-mix.md)。

跑通之后还会关心两件事，都在工程化段：**换个 shape 就重新编译怎么办**（[JIT 参数与编译缓存](./jit-arguments.md)）、**生成的代码到底长什么样**（[编译选项与产物观察](./compiler-options.md)）。
