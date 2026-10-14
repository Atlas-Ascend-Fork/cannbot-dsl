# 调试与精度验证

算子开发的时间大头通常不在写代码，而在「为什么结果不对」和「为什么不够快」。这一节给出排查顺序和可用工具。

## 排查顺序

```text
编译不过？        → 看报错类别（构图期 / 后端），对照控制流和资源约束
      ↓
生成的不对？      → CANNBOTDSL_DUMP_ASCENDC=1，直接读 .asc 看生成了什么
      ↓
结果全错？        → 先确认 Host 侧 tiling 和分核逻辑，再看搬运方向
      ↓
部分元素错？      → 看尾块、掩码、对齐
      ↓
偶发 / 多核才错？ → 看同步和 Cache 一致性
      ↓
精度差一点？      → 看累加顺序、中间精度、量化
      ↓
结果对但慢？      → 见《高性能算子编写指南》
```

## 构图期调试：`print`

最便宜的调试手段。Python 内建 `print` 在构图期执行，用来确认「我生成的是什么样的 kernel」：

```python
@kernel
class MyKernel:
    def __init__(self, num_col, dtype):
        self._tile = compute_tile(num_col, dtype)
        self._branch = pick_branch(num_col)
        print(f"[构图] num_col={num_col} tile={self._tile} branch={self._branch}")
        ...
```

能在构图期发现的问题，不要留到设备上去发现：tile 算成 0、分支选错、循环次数为负、UB 预算算超。

## 设备侧调试：`cbd.print`

```python
cbd.print(value, *, label=None, mode="ring")
```

在 Kernel 中打印 Tensor 或标量，输出到当前进程的 `stdout`。

```python
import cannbotdsl as cbd


@cbd.kernel
def debug_kernel(src: cbd.Tensor):
    tmp = cbd.Channel(cbd.MemLoc.UB, shape=(16,), dtype=src.dtype, depth=1)
    slot = tmp.produce()
    cbd.mem_copy(slot, src)
    slot = tmp.consume()
    cbd.print(slot, label="ub_data")        # 打印 UB 上的整个 Tensor
    cbd.print(src[(0,)], label="first")     # 打印 GM 上点索引得到的标量
```

### 支持范围

| 输入 | 支持 |
| --- | --- |
| Tensor | UB、L1、L0C |
| 标量 | Kernel 局部标量，以及从 GM / UB Tensor 点索引得到的标量；bool、8/16/32/64 位整数、float16、bfloat16、float32 |

### 约束

- 必须在 `@kernel` 函数中调用，不能直接打印 Python 数值或 `torch.Tensor`。
- **Channel 不能直接打印**，要先 `produce()` / `consume()` 取出 Tensor。
- Tensor 按物理存储顺序采样，**最多 1024 字节**。非连续视图不会自动压紧，NZ/ZN 数据也不会自动重排为 ND——打印分形数据时看到的顺序是物理顺序。
- 每个打印点在每个 block/subblock 上独立保留 **64 条**记录。`mode="ring"` 保留最近 64 次，`mode="stop"` 保留最早 64 次（只停止保存新数据，不停止 Kernel 执行）。
- **会增加设备内存占用并触发 Host 回读和同步，不能用于性能测量。**

### 结构化记录

除文本输出外，Host 侧可以拿到结构化记录：

```python
from cannbotdsl.core.diag.debug import clear_debug_prints, get_debug_prints

clear_debug_prints()
run(src)
torch.npu.synchronize()

records = get_debug_prints()
for r in records:
    if r["label"] == "ub_data":
        print(r["shape"], r["data"])
```

每条记录包含标签、dtype、shape、存储位置、执行序号、block/subblock 信息。这让你可以在 Host 侧**自动比对**设备中间结果和参考实现，比肉眼看 stdout 高效得多。

典型用法：在 kernel 的每个阶段末尾打一个带标签的点，Host 侧逐阶段和 numpy/torch 参考实现对比，定位第一个出错的阶段。

## 看寄存器：`dump_reg`

`cbd.print` 只支持 UB / L1 / L0C 上的 Tensor，**看不到矢量寄存器**。而 Reg 编程模型的全部价值就是「中间结果留在寄存器里」——最想看的那一段恰好是盲区。

补上这个洞的是 `dump_reg`：

```python
cbd.dump_reg(register, *, desc=None, dump_size=None, dtype=None)
cbd.dump_tensor(tensor, *, desc=None, dump_size=None, offset=0, shape=None, dtype=None)
```

| 接口 | 看什么 | 在哪调用 |
| --- | --- | --- |
| `dump_reg` | **矢量寄存器的 lane 内容** | 必须在 `with vf(mode="simd")` 内，对 `RawVReg` 调用 |
| `dump_tensor` | Tensor 的原始字节 | `@kernel` 内 |

```python
with vf(mode="simd"):
    acc = vadd(vload(x, 0), vload(y, 0), mask=m)
    cbd.dump_reg(acc, desc="after_vadd")      # 看这一拍 64 条 lane 的值
    vstore(out, 0, acc, m)
```

::: warning 和 `cbd.print` 的区别
`dump_reg` / `dump_tensor` 走的是 CANN native dump 流，**不会**出现在 `get_debug_prints()` 的结构化记录里，输出要从 CANN 的 dump 通道取。单次 VF 的载荷区满了之后会丢弃后续记录，下次 VF 清除溢出状态。

它们占用的正是 UB 顶部那 2 KB native dump 预留区。
:::

::: info 包内可用 · 文档站未收录
两个接口在包顶层导出（`cbd.dump_reg` / `cbd.dump_tensor`），实现在 `cannbotdsl/core/diag/native_dump.py`，官方 API 文档站只列了名字。
:::

## 精度验证

### 测试组织

测试目录与 `samples/` 对应。先装依赖：

```bash
python -m pip install -r requirements.txt
```

运行单个样例的测试：

```bash
python3 -m pytest test/sparse_flash_attention/test_sparse_flash_attention.py -v
```

修改一个样例时，优先运行与它直接对应的测试；提交前再按 CI 要求扩大范围。

### 写测试的要点

```python
import pytest
import torch
import torch_npu  # noqa: F401

import cannbotdsl


@pytest.mark.parametrize("n", [64, 1000, 65536, 1048577])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_my_op(n, dtype):
    cannbotdsl.clear_compile_cache()      # 从干净状态开始

    x = torch.randn(n, dtype=dtype).npu()
    out = my_op(x)
    torch.npu.synchronize()               # 必须同步后再读结果

    golden = reference(x.float()).to(dtype)   # golden 用 fp32 算
    torch.testing.assert_close(out.cpu(), golden.cpu(), rtol=1e-3, atol=1e-3)
```

| 要点 | 说明 |
| --- | --- |
| **必须 `torch.npu.synchronize()`** | 不同步就读结果会拿到未完成的数据 |
| **golden 用高精度算** | bf16/fp16 输入先转 fp32 算参考值，再转回比较 |
| **覆盖边界 shape** | 整除、带尾块、单 tile、小于核数、超大 |
| **覆盖所有支持的 dtype** | 不同 dtype 走的是不同的生成代码 |
| **容差要有依据** | 参考仓库做法，例如 matmul 用相对误差 `< 1e-2 * max(|ref_max|, 1.0)` |

### 精度不对时的四个嫌疑人

按从「最容易被忽略」到「最容易想到」排序：

**1. 非规格化数被刷成零（FTZ）。** 3510 硬件不实现非规格化数，`--cce-ftz` 默认开启 flush-to-zero，详见[编译选项与产物观察](/zh/guide/programming-model/compiler-options#cce-ftz-非规格化数的处理)。

::: danger 这类问题的症状很有误导性
表现是**个别元素误差极大**，而不是整体误差偏大——非常像掩码越界或搬运方向错。

**怎么快速排除它**：把 golden 里绝对值极小（落在非规格化区间）的元素单独挑出来，看设备侧是不是正好 0。是的话就是 FTZ，不用再去查搬运。
:::

**2. 累加顺序。** 设备上按 tile 顺序累加，golden 如果用 `torch.matmul` 一次算完，浮点舍入的累积方式不同。golden 一律用 fp32 累加，并按算子实际容差比对。

**3. 量化本身的误差。** MX 格式每 32 个元素共享一个 2 的幂次 scale，这是有损的。参考实现必须用**同样的块长和同样的 scale 取整方式**，否则比的是两种量化方案，和算子实现无关。见[数据类型与量化](/zh/guide/programming-model/data-types)。

**4. 中间精度。** bf16/fp16 输入按 fp32 算中间结果再转回，和全程 16 位算，结果不同。确认设备侧和 golden 用的是同一种策略。

### 独立的 checker

复杂算子的参考实现往往本身就不简单。仓库的做法是放一个独立的 checker 文件：`sparse_flash_attention_validation.py`、`qsa_indexer_checker.py`、`flash_attn_fp8_fullquant_checker.py` 等。把参考实现和测试驱动分开，参考实现可以复用，也更容易单独验证其正确性。

## 常见错误与定位

### 编译期错误

| 现象 | 常见原因 |
| --- | --- |
| 「运行时数据不能传给只在编译期处理的接口」 | 把运行期值传给了 `const_expr()` / `range_constexpr()` |
| 变量在控制流外不可用 | 在 `if` / `for` 内部创建的 Tensor 或变量带出了作用域 |
| 类型不兼容 | 同一变量在不同分支/迭代中类型不一致 |
| 不支持的搬运组合 | 源/目标的存储位置、dtype、布局不匹配，或 engine 选项与方向不兼容 |
| 片上存储超限 | tile 太大，UB / L1 / L0 放不下 |
| `break` / `continue` / `return` 报错 | 设备循环不支持提前退出 |

Host 侧的构图、编译和运行错误由 `CANNBotError` 及 `DiagnosticCode` 表达，见 [Host API](/zh/api/host/)。

### 结果错误

| 现象 | 优先检查 |
| --- | --- |
| 全部为 0 或未初始化值 | 输出 Tensor 没搬回 GM；或 `produce()` / `consume()` 配对错了 |
| 只有第一个核的结果对 | 分核逻辑：`get_block_idx()` 没参与 tile 坐标计算 |
| 最后几个元素错 | 尾块：掩码没覆盖到，或 `tile_slice` 的 coord 越界 |
| 矩阵乘结果错位 | 分形对齐：M/N 没对齐 16，K 没对齐 K0；或 transpose 配置与实际布局不符 |
| 累加结果偏小/偏大 | `init` 参数：第一块应为 `True`，后续应为 `False` |
| 掩码位置的值是垃圾 | 掩码寄存器未经 `full_mask` / `create_mask` / `update_mask` 赋值就使用 |
| 多核运行偶发错误 | 核间同步缺失；或 Scalar 写 GM 后未处理 DCache 一致性 |
| 多核写同一区域结果随机 | 地址未对齐到 Cache Line，需要用 `*_bypass` 接口绕过 DCache |

### 同步与 Cache 问题的排查

多核偶发错误最难查。按这个顺序试：

1. 把 `block_dim` 降到 1，看是否还错。还错 → 不是并发问题。
2. 在可疑的写入之后加 `global_sync_all()`，看是否变正确。变正确 → 缺同步。
3. 检查 Scalar 对 GM 的读写是否需要 `dcci_*` / `dci`，或改用 `load_bypass` / `vec_store_bypass` / `cube_store_bypass`。
4. 检查多核写入的 GM 地址是否对齐到 Cache Line。

## 性能分析

### 用 msprof 采集

仓库样例的口径统一是「msprof 采集 Task Duration，每 shape 重复 10 次取 min」。这比 Python 侧计时准确——Python 计时会混入 Host 侧开销和 stream 排队。

**完整的测量规范**（预热次数、采样次数、取值方式、去离群、环境固定）在[高性能算子编写指南 · 测量规范](/zh/guide/programming-model/performance#测量规范)，这里不重复。三条最容易漏的：

- 取**最小值**，不是平均值。
- `torch.npu.synchronize()` 放在循环**外**。
- 测量前关掉所有 `cbd.print` / `dump_*` 和 `CANNBOTDSL_DUMP_*`——前者触发回读与同步，后者旁路编译缓存强制重编译。

### Python 侧粗测

快速判断量级时可以这样做，但要注意预热和同步：

```python
my_op(x)                      # 预热，触发 JIT 编译
torch.npu.synchronize()

t0 = time.perf_counter()
for _ in range(100):
    my_op(x)
torch.npu.synchronize()       # 关键：循环外同步一次
dt = (time.perf_counter() - t0) / 100
```

### DSL 内置的 profiling 入口

Host API 提供 `ProfileSpec` 和 `profiler.report_tensor_info` 用于配置 profiling 与报告张量信息，见 [Host API](/zh/api/host/)。

设备侧可以用 [`get_system_cycle()`](/zh/api/kernel/system/get-system-cycle.html) 读周期数，给关键代码段做粗粒度计时。注意这会引入标量指令，别放在最内层循环里。

### 对比基线

判断「够不够快」要有参照：

- **和 CANN 内置算子比**：`torch_npu` 的对应算子。
- **和理论上限比**：访存受限算子对带宽，计算受限算子对 MAC 峰值。
- **和仓库样例比**：各样例 README 里有性能对比图和加速比区间，例如 matmul_streamk 在 100 个纯 SK 用例上 geomean 加速比 0.968、85 个用例 ≥ 0.9。

## 一份实用的调试清单

开始查错之前先确认这些：

- [ ] `torch.npu.synchronize()` 调了吗
- [ ] 输入 Tensor 都在 NPU 上、都是 contiguous 吗
- [ ] 输出 Tensor 分配了吗、dtype 对吗
- [ ] `block_dim` 是不是 0 或者超过了可用核数
- [ ] tile 大小是不是算成了 0 或负数（构图期 `print` 一下）
- [ ] 每个 `produce()` 都有对应的 `consume()` 吗
- [ ] 是不是在 VF 内调用了 `produce()` / `consume()`
- [ ] 掩码在使用前赋值了吗
- [ ] K 方向累加的第一块是不是 `init=True`

## 下一步

[AOT 与 Native 算子包](/zh/guide/programming-model/aot-packaging)：调通之后怎么固化成可发布的产物。
