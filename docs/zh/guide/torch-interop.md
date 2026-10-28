# torch 接口与 stream 语义

前面各章讲的都是 Kernel 以内的事。这一章讲最外面那一层：**普通 Python 和 PyTorch 怎么把数据交给 DSL，又怎么安全地把结果取回来。** 这一层不受 DSL 约束，但它决定了你的算子能不能被别人用。

## 三层结构里的第 3 层

```text
┌──────────────────────────────────────────────────────┐
│ 第 3 层：torch 接口函数（普通 Python）     ← 本章      │
│   校验 shape/dtype、分配输出与 workspace、reshape、   │
│   决定 block_dim、调用 @host                          │
├──────────────────────────────────────────────────────┤
│ 第 2 层：Host 侧 tiling + @host 入口                  │
├──────────────────────────────────────────────────────┤
│ 第 1 层：@kernel 设备代码                             │
└──────────────────────────────────────────────────────┘
```

这一层是**纯 Python**，可以随意写校验、异常、reshape、padding。`samples/rms_norm/rms_norm.py` 的 `rms_norm()` 是一个完整参考。

## Tensor 要满足什么

传给 `@host` 函数的 Tensor 必须是真实的 NPU 张量：

```python
import torch
import torch_npu  # noqa: F401   # 向 PyTorch 注册 Ascend NPU 后端
```

`import torch_npu` 这一行不能省——它是注册后端的副作用导入，少了它 `device="npu:0"` 不存在。

| 要求 | 为什么 | 不满足怎么办 |
| --- | --- | --- |
| 在 NPU 上 | Kernel 直接按 GM 地址访问 | `.npu()` 或 `device="npu:0"` |
| `contiguous` | 多数搬运路径假设紧凑 stride | `.contiguous()`，或用 `Dim` 声明动态 stride |
| dtype 在支持集内 | 不同 dtype 走不同生成代码 | 在接口层抛清晰异常 |
| 输出已分配 | Kernel 不返回值 | `torch.empty_like(x)` 等 |

## dtype 映射

样例各自维护一份映射字典，建议统一成这样：

```python
_TORCH_TO_DSL = {
    torch.float16: dtypes.float16,
    torch.bfloat16: dtypes.bfloat16,
    torch.float32: dtypes.float32,
    torch.int8: dtypes.int8,
    torch.int32: dtypes.int32,
}
```

::: info 待核实
低精度类型（FP8 的两种、FP4、E8M0）在 `cbd.dtypes` 下的确切名字请以 [API 接口清单](../api/language/decorators.md)为准。torch 侧对应的是 `torch.float8_e4m3fn`、`torch.float8_e5m2` 等；打包的 FP4 在 torch 侧通常以 `uint8` 承载，由算子接口按约定解释。格式层面的说明见[数据类型与量化](./data-types.md)。
:::

## stream 语义：为什么必须 synchronize

这是最容易踩的坑，而且踩了不报错、只是数据不对。

**Kernel 的启动是异步的。** `run(...)` 返回时，设备上的计算很可能还没开始，更不用说结束。它只是把任务下发到了 stream 上：

```text
Host 线程：  run(x, y, out) ──► 下发到 stream ──► 立即返回
                                      │
设备：                                 └─► 排队 ──► 执行 ──► 完成
```

所以：

```python
run(src0, src1, dst)
torch.npu.synchronize()          # ← 必须。等 stream 上的任务全部完成
print(dst.cpu())                 # 现在读才是对的
```

| 什么时候必须同步 | 原因 |
| --- | --- |
| 读结果之前（`.cpu()`、`.item()`、断言） | 否则读到未完成的数据 |
| 计时之前与之后 | 否则测到的是下发时间 |
| 调 `get_debug_prints()` 之前 | 设备打印需要回读 |

什么时候**不需要**手动同步：

- 后续操作也在同一条 stream 上（例如紧接着一个 torch 算子）。stream 内天然有序。
- `.cpu()` 这类拷贝操作本身通常隐含同步——但**不要依赖它**，显式写出来更可靠。

::: warning 计时一定要把同步放在循环外
```python
my_op(x)                      # 预热，触发 JIT 编译
torch.npu.synchronize()

t0 = time.perf_counter()
for _ in range(100):
    my_op(x)
torch.npu.synchronize()       # 关键：循环外同步一次
dt = (time.perf_counter() - t0) / 100
```
在循环内同步会把流水打断，测出来的数会偏大。而不同步就只测到了下发开销，数会偏小得离谱。
:::

AI CPU 的调用也在同一条 stream 上，这正是 metadata 与 AI Core kernel 之间不需要额外同步的原因。见 [AI CPU 与调度计划](./aicpu.md)。

## 输出的三种形态

Kernel 不返回值，所有输出都通过参数传入。但对外的 torch 接口可以是任意形态：

### 1. 单输出

```python
def vec_add(x, y):
    out = torch.empty_like(x)
    VecAdd().run(x, y, out)
    return out
```

### 2. 多输出

`rms_norm` 返回 `(y, rstd)`，`flash_attn` 返回 `(output, softmax_max, softmax_sum)`。做法一样——在接口层分配好全部输出，一起传进去，最后按元组返回：

```python
def rms_norm(x, gamma, epsilon=1e-6):
    y = torch.empty_like(x)
    rstd = torch.empty(rows, 1, dtype=torch.float32, device=x.device)
    RmsNorm(dtype).run(x2d, gamma2d, y2d, rstd, float(epsilon))
    return y, rstd
```

### 3. 原地更新

`kv_compress_epilog` 把量化结果按 slotMapping 散写进 cache 并**原地更新**；`fused_recurrent_kda` 原地更新 state pool。这类算子的约定是：

- 被原地修改的张量**既是输入也是输出**，在文档和签名里要写清楚。
- 返回值通常把它一并返回，便于链式调用：`return (state, out)`。

::: danger 输入输出别名要显式确认
把同一个张量同时当输入和输出（`out is x`）是否安全，取决于 Kernel 内部的搬运顺序是否存在读写重叠。**默认不要假设安全。** 需要原地的算子应当在设计时就明确支持，并在测试里专门覆盖这个用法。
:::

### Tensor 列表

`grouped_matmul` 的输入输出是张量列表，返回 `List[Tensor]`。提前编译时对应的参数规格是 `TensorListSpec`，它描述列表长度以及每个元素的类型与形状规则；列表长度也可以用 `Dim` 声明成动态的。见 [JIT 参数与编译缓存](./jit-arguments.md)。

## workspace 的分配

Stream-K、FlashDecode 这类算法需要一块 GM 暂存部分和。**它就是一个普通 torch 张量，在这一层分配**：

```python
def matmul_streamk(a, b):
    c = torch.empty(m, n, dtype=a.dtype, device=a.device)
    # 按 tiling 推出需要多少行
    ws_rows = tiling.sk_tiles * tiling.sk_splits
    workspace = torch.empty(ws_rows, n, dtype=torch.float32, device=a.device)
    MatmulStreamK(tiling).run(a, b, c, workspace)
    return c
```

要点：

- **容量由 tiling 推导**，不要给一个「够大的」固定值——它会白占显存。
- **dtype 通常是 fp32**，因为存的是待归约的部分和，精度不能降。
- **要不要清零取决于算法**。如果每个核都会完整写自己那一段，就不用清；如果用 `atomic_add` 累加，必须先清零。
- 允许调用方传入自己的 workspace（便于跨调用复用），但要校验。`flash_attn` 的 `validate_workspace()` 检查 ndim、宽度、行数下界、dtype、device 与连续性，这个做法值得照搬。
- workspace 的**阶段间同步要自己写**，Channel 不管它。见[同步、Cache 与跨核交接](./synchronization.md)。

## 非整除长度：在这一层对齐

`tile_slice` 与 Channel 槽位的 shape 必须一致，所以**不要把尾块缩成 `(actual,)` 再拷进 `(tile,)` 的槽**。正确做法是在这一层把 GM 对齐，设备侧用掩码屏蔽多余 lane：

```python
n = x.numel()
n_tiles = math.ceil(n / tile)
n_pad = n_tiles * tile

x1d = x.reshape(n)
if n_pad != n:
    x1d = torch.nn.functional.pad(x1d, (0, n_pad - n))
    o1d = torch.empty(n_pad, dtype=x.dtype, device=x.device)
else:
    o1d = out.reshape(n)

VecAdd().run(x1d, y1d, o1d, block_dim)

if n_pad != n:
    out.copy_(o1d[:n].reshape_as(out))
```

## 封装成 torch 自定义算子

仓库样例对外暴露的是**普通 Python 函数**。这对脚本和测试够用，但要在模型里用、或者走 `torch.compile` 的图模式，就需要注册成自定义算子。

::: info 待核实
CANNBot-DSL 0.7.0 没有提供专门的 torch 算子注册辅助接口，仓库样例也没有这么做。下面给出的是 PyTorch 通用路径，可行性需要你在目标版本上实际验证，本文档不作保证。
:::

通用做法是用 `torch.library` 注册，并补一个 meta 实现（只推形状、不算数值），让 `torch.compile` 和 fake tensor 能走通：

```python
import torch

@torch.library.custom_op("mylib::vec_add", mutates_args=())
def vec_add_op(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return vec_add(x, y)          # 内部调用 @host

@vec_add_op.register_fake
def _(x, y):
    return torch.empty_like(x)    # 只描述输出形状与 dtype
```

原地更新的算子要在 `mutates_args` 里声明被修改的参数名，否则图优化可能把它重排掉。

## 单次调用开销

即使 Kernel 本身很快，这一层也有固定开销：参数校验、dtype 查表、reshape、Tensor 到 DSL 的转换、缓存查找。小算子上这部分可能和设备执行时间同一量级。

降低办法：

| 手段 | 效果 |
| --- | --- |
| 用 `cbd.compile(...)` 提前编译并复用 `program` | 跳过每次的缓存查找与构图 |
| 把编译产物缓存在模块级变量里（配锁） | 全进程只编译一次，见 `flash_kda` 的写法 |
| 用 `Dim` 声明高频变化的维度 | 避免每换一个 shape 就重新编译 |
| 部署走 [AOT](./aot-packaging.md) | 彻底消除首次编译开销 |
| 把校验收敛成一次 | 不要在热路径上重复 `is_contiguous()` 这类检查 |

## 检查清单

| 项 | 检查 |
| --- | --- |
| `import torch_npu` | 写了吗（即使看起来没用到） |
| 设备 | 输入都在 NPU 上吗 |
| 连续性 | `is_contiguous()` 查了吗，不连续时的路径定了吗 |
| dtype | 在支持集内吗，不支持时抛的异常清晰吗 |
| 输出 | 分配了吗，dtype 与 shape 对吗 |
| workspace | 容量按 tiling 推的吗，要不要清零 |
| `block_dim` | 是 `min(需要的核数, 可用核数)` 吗 |
| 对齐 | 非整除长度在这一层 pad 了吗 |
| 同步 | 读结果前 `torch.npu.synchronize()` 了吗 |
| 计时 | 预热了吗，同步在循环外吗 |
| 原地 | 被修改的张量在签名和文档里写明了吗 |

## 下一步

[编译选项与产物观察](./compiler-options.md)：哪些开关会影响生成的代码，以及怎么把它 dump 出来看。
