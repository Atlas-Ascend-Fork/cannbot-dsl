# mega_mla

## 产品支持情况

- Ascend 950（NPU ARCH 3510）。
- Query head 数 `H=8/16/64`，head dimension 为 512；hidden size 为 5120，Q LoRA rank 为 1280。
- 支持多请求 decode，Query 可按等长方式展平为 `T=B*S`，也可通过 `cu_seqlens_qo` 组织为 packed Query，支持空请求。
- `grid` 支持从 1 到当前流可用 AIC/AIV 配对核数的任意正整数，使用对应的 `2*grid` 个 AIV。

## 功能说明

基于 CANNBotDSL 的 DeepSeek V4.1 attention 融合算子，将 Q/KV 投影、RMSNorm、
RoPE、window cache 更新和双路稀疏 Attention 合并为一次 mixed AIC/AIV kernel 调用。

输入为投影前的 hidden states，算子先生成 Q 和 KV，将当前 KV 写入 window cache，
再按索引从 window cache 和压缩 KV cache 读取参与 Attention 的记录。两路解码后的
KV 同时用作 Key 和 Value，`attn_sink` 仅参与 Softmax 分母，输出为 BF16 `[T,H,512]`。

令 `q` 为归一化、RoPE 和缩放后的 Query，`v_j` 为选中的解码 KV，则数学表达为：

```text
s_j = q · v_j
Z = exp(attn_sink) + sum_j exp(s_j)
out = sum_j exp(s_j) * v_j / Z
```

实现使用稳定 Softmax、FP32 累加与归一化，并在中间阶段和输出处按 BF16 精度舍入。
无有效 KV 或无效 Query 的输出为零。

| 特性 | 支持范围 |
| :--- | :--- |
| 输入 / 输出 | BF16；输入 `[T,5120]`，输出 `[T,H,512]` |
| 投影权重 | E4M3 NZ 存储；每 32 个值共用一个 E8M0 scale |
| KV cache | window：FP8、544 B/记录；压缩：FP4、320 B/记录 |
| 索引 | 物理索引，或两路独立页表对应的请求内逻辑索引 |
| 请求组织 | 等长展平或 packed Query，支持空请求和有效长度 |
| Query 元数据 | 可在同一推理步内导出并跨层复用 |

## 接口与参数

公共入口为 [`mega_mla()`](mega_mla.py)，直接传入业务张量。
所有张量位于同一
NPU；除 NZ 权重和下述原生 cache 页间 stride 外，均采用连续 ND 布局。
`T` 为展平后的 Query token 数，`Pw/Pc` 分别为 window/压缩 cache 的页大小，
`Kw/Kc` 为两路索引列容量，`K=Kw+Kc`；`Nw/Nc` 为两路 cache 的物理页数。

设备主入口为 `MegaMlaKernel.mega_mla`（`@kernel`）。
提供 seqlens 时使用 `mega_mla_with_seqlens`；
Query 元数据导出或复用时使用 `mega_mla_with_metadata`。

### 调用配置

```python
out = mega_mla(**inputs, cmp_decode_strategy="auto")
```

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `grid` | `None` | 默认使用当前流允许的 AIC/AIV 配对核数；可显式指定正整数且不超过该上限，不要求偶数或整除 T |
| `cmp_decode_strategy` | `"auto"` | 压缩 cache 的读取策略，见实现说明 |

`H` 从 `attn_sink`、`T` 从 `hidden_states`、`Pw/Pc` 从各自 cache、`Kw/Kc` 从两路索引
shape 推导，无需构造算子对象。支持 `H=8/16/64`；`Kw` 须同时为 64 和 Pw 的倍数。
计数参数须为正整数，`T>0`、`Kc>0`、`K%128=0`，`T` 无需整除设备核数。
`grid` 表示参与本次调用的 AIC 数量，AIV 按 1:2 配对；每次启动前通过 get_platform_info(stream=...) 刷新可用核数；超过 AIC 或 AIV 配对上限时，
在分配 workspace 和编译前报错。P1 的 20 个 Q 投影任务与 8 个 KV 投影任务由参与核
循环领取，少于 28 核也会完整计算；各阶段同步仅覆盖本次启动的核。
计算模式固定为 attention，RoPE 维度固定为 64。底层类接口为 `MegaMlaOp`。

### 必选输入

下表顺序与函数的位置参数一致。`>=T` 表示允许预分配更多行，实际仅读取前 T 行。

| 参数 | shape | dtype / 布局 | 说明 |
| :--- | :--- | :--- | :--- |
| `hidden_states` | `[T,5120]` | `bfloat16` | 输入激活 |
| `query_norm_weight` | `[1,1280]` | `float32` | Q RMSNorm 权重 |
| `kv_norm_weight` | `[1,512]` | `float32` | KV RMSNorm 权重 |
| `query_down_proj_weight` | 逻辑 `[1280,5120]` | `float8_e4m3fn`，NZ | WqA 权重 |
| `kv_proj_weight` | 逻辑 `[512,5120]` | `float8_e4m3fn`，NZ | WKV 权重 |
| `query_up_proj_weight` | 逻辑 `[H*512,1280]` | `float8_e4m3fn`，NZ | WqB 权重 |
| `query_down_proj_scale` | `[1280,80,2]` | `float8_e8m0fnu` | WqA MX scale |
| `kv_proj_scale` | `[512,80,2]` | `float8_e8m0fnu` | WKV MX scale |
| `query_up_proj_scale` | `[H*512,20,2]` | `float8_e8m0fnu` | WqB MX scale |
| `k_cache` | `[Nw,Pw,1,544]` | `uint8`，页内连续 | 可读写的 window cache |
| `extra_k_cache` | `[Nc,Pc,1,320]` | `uint8`，页内连续 | 只读的压缩 KV cache；原生布局 Pc 支持 64/128 |
| `indices_in_kvcache` | `[>=T,Kw]` | `int32` | window 索引 |
| `extra_indices_in_kvcache` | `[>=T,Kc]` | `int32` | 压缩 KV 索引 |
| `attn_sink` | `[1,H]` | `float32` | 每个 head 的 sink 分数 |
| `rope_cos`、`rope_sin` | 各 `[>=T,32]` | `float32` | 当前 token 已选好的 32 对 RoPE 系数 |
| `window_cache_slots` | `[>=T]` | `int32` | 当前 KV 写入 window cache 的物理行号 |

三份投影权重由调用方在推理循环外使用
`torch_npu.npu_format_cast(weight, torch_npu.Format.FRACTAL_NZ)` 转换一次。
输入 `weight` 应为 NPU 上的 E4M3 ND 张量；转换后逻辑 shape 和 E4M3 dtype 不变，
以返回的 NZ 根张量作为算子入参。已准备好的 E4M3 NZ 权重可直接传入。
scale 保持表中的连续行主序布局，不随权重转换为 NZ。

### 原生 cache 布局

两路 cache 的页大小和页间 stride 独立，从原始 Tensor 获取，无需额外传入标量参数：

| Cache | shape | stride（UINT8 元素，即字节） |
| :--- | :--- | :--- |
| window | `[Nw,Pw,1,544]` | `[Sw,544,544,1]`，`Sw = k_cache.stride(0)` |
| 压缩 KV | `[Nc,Pc,1,320]` | `[Sc,320,320,1]`，`Sc = extra_k_cache.stride(0)` |

页内记录必须紧密排列，页间允许 padding；`Sw >= Pw*544`，`Sc >= Pc*320`，
不允许页相互重叠。支持原生 Tensor 的非零 storage offset。
例如同样的 window `Pw=128`，不同层的压缩 cache 可以分别是 `Pc=128` 或 `Pc=64`，
它们的 `Sc` 也可不同。算子不执行 `.contiguous()` 或 cache 重打包。

逻辑索引先用对应页大小查页表，再按对应页 stride 寻址：

```text
physical_page = table[batch, index // P]
row_in_page  = index % P
address      = cache.data_ptr() + physical_page * S + row_in_page * record_bytes
```

window 代入 `P=Pw,S=Sw,record_bytes=544`；压缩侧代入 `P=Pc,S=Sc,record_bytes=320`。
`data_ptr()` 已包含 view 的 storage offset，不再重复加该偏移。
未提供页表时，`index // P` 直接是物理页号。
编译缓存包含两路原始 cache 的 shape 和 stride；同一算子实例切换页布局会选择不同编译产物。

### 可选参数与返回值

| 参数 | shape / 类型 | 说明 |
| :--- | :--- | :--- |
| `out` | `[T,H,512]`，`bfloat16` | 可选的预分配输出；须连续且与输入同设备 |
| `topk_length` | `int32[N]` | window 索引的有效前缀列数；省略时使用完整列容量 `Kw` |
| `extra_topk_length` | `int32[N]` | 压缩 KV 索引的有效前缀列数；省略时使用完整列容量 `Kc` |
| `block_table` | `int32[B,Lw]` | window 逻辑页到物理页的映射；须与 `extra_block_table` 同时提供或省略 |
| `extra_block_table` | `int32[B,Lc]` | 压缩 KV 逻辑页到物理页的映射；须与 `block_table` 同时提供或省略 |
| `cu_seqlens_qo` | `int32[B+1]` | packed Query 的请求边界，允许空请求 |
| `seqused_q` | `int32[B]` | 每个请求实际有效的 Query 前缀长度 |
| `cache_seqlens` | `int32[B]` | window 有效 KV 逻辑记录数，须配合 `block_table` |
| `extra_cache_seqlens` | `int32[B]` | 压缩 KV 有效逻辑记录数，须配合 `extra_block_table` |
| `sequence_metadata` | `AttentionSequenceMetadata` | 同一推理步内其他层导出的 Query 元数据 |
| `return_sequence_metadata` | `bool`，默认 `False` | 为真时同时返回 Query 元数据 |

默认返回 `[T,H,512]` BF16 张量，并原地更新 `k_cache`。提供 `out` 时返回该对象；
省略时分配独立输出。设置 `return_sequence_metadata=True` 时返回 `(out, metadata)`。

### 索引与长度约束

- 未提供页表时，两路索引分别为各自 cache 的物理行号 `page*P+offset`，P 取对应 Pw/Pc。
  负索引和有效前缀之外的列被屏蔽；其余物理索引须在对应 cache 容量内。
- 提供页表时，索引为请求内逻辑记录号，物理行号为
  `block_table[b,index//P]*P+index%P`，两路分别使用 Pw/Pc。
  两路索引独立编号；压缩侧使用压缩后的记录号，页 stride 不参与逻辑行号计算。
  负索引、越界逻辑索引和无效物理页号被屏蔽。
- `window_cache_slots` 始终为 window 物理行号。值小于 Pw 时跳过写入，其余值须小于
  `Nw*Pw`；同次调用的有效写入位置应互不冲突，写入地址采用 Sw。
- 未提供 packed 边界时，Query 按 `T=B*S` 等长展平，要求 `T%B=0`。
  `cu_seqlens_qo` 须从 0 开始、以 T 结束且非递减；`seqused_q[b]` 不得超过对应 span。
- 启用序列长度或 Query 元数据接口时，TopK 长度可以分别为 `[B]` 或 `[T]`；
  否则要求 `N>0`、`T%N=0`，两路均提供时 shape 相同，token t 使用第 `t//(T/N)` 项。
  两路长度值分别位于 `[0,Kw]` 和 `[0,Kc]`。
- `cache_seqlens` 限制可访问的 KV 记录范围，TopK 长度限制索引列数，二者含义独立。
  KV 长度不得超过对应逻辑页表容量；省略时使用该容量。
- 无效 Query 输出为零且不写 cache。算子不额外生成因果 mask，时序候选由业务索引决定。
  host 仅检查张量元数据，调用方应保证设备侧索引、长度和写入位置满足上述数值约束。

## 快速开始

### Native wheel

本算子已登记到统一编包工程的 `ds41` 组，组构建命令为 `./run-build.sh --group ds41`。
安装后从 `ops.mega_mla` 导入 `mega_mla`。默认预编译配置和自定义 profiles 字段见
[Native 导出入口](mega_mla_native.py)；通用构建流程和二进制命中模式见
[Native 编包说明](../../../packaging/native/README.md)。

需要支持 Layout 资源接口的 CANNBotDSL、PyTorch 2.7 及以上、配套 torch_npu 和 CANN。
PyTorch 必须同时提供 float8_e4m3fn 和 float8_e8m0fnu；E8M0 用于三路投影 scale。
权重须为 FRACTAL_NZ 格式、零 storage_offset、稠密逻辑布局并覆盖完整存储；同形状的完整存储别名可复用。
资源使用 `Buffer/RingBuffer(mem_loc, dtype, make_layout(shape), ...)` 声明。
发行包从 [PyPI 公开发行页](https://pypi.org/project/cannbot-dsl/) 获取，安装步骤见
[仓库说明中的安装章节](../../../README.md#2-安装-cannbot-dsl)。
安装好配套依赖后，从仓库根目录加载环境并设置模块路径：

```bash
source "${install_path}/ascend-toolkit/set_env.sh"
export PYTHONPATH="$PWD/samples/attention/mega_mla:${PYTHONPATH:-}"
```

将下列示例保存为 `mega_mla_example.py`，在上述支持的设备上执行。
示例使用非零激活和常数权重：每个 Query 只选择本次写入的 window KV，压缩侧索引全部
屏蔽。归一化后的 KV 为全 1，输出应接近全 1；末尾断言同时检查读写链路和计算结果。
业务中应替换为实际量化权重、编码后的 cache、索引和位置系数，不能用浮点到 UINT8 的
数值转换代替 cache 编码。

```python
import torch
import torch_npu

from mega_mla import mega_mla

torch.npu.set_device(0)
device = torch.device("npu", 0)
T, H, P = 8, 8, 128

# 在 CPU 初始化 E4M3 常数权重，搬到 NPU 后在推理循环外转为 NZ。
wa = torch_npu.npu_format_cast(
    torch.full((1280, 5120), 1 / 64, dtype=torch.float32).to(torch.float8_e4m3fn).to(device),
    torch_npu.Format.FRACTAL_NZ,
)
wkv = torch_npu.npu_format_cast(
    torch.full((512, 5120), 1 / 64, dtype=torch.float32).to(torch.float8_e4m3fn).to(device),
    torch_npu.Format.FRACTAL_NZ,
)
wb = torch_npu.npu_format_cast(
    torch.full((H * 512, 1280), 1 / 64, dtype=torch.float32).to(torch.float8_e4m3fn).to(device),
    torch_npu.Format.FRACTAL_NZ,
)


def unit_mx_scale(shape):
    # E8M0 编码 127 表示 scale=1。
    return torch.full(shape, 127, dtype=torch.uint8, device=device).view(
        torch.float8_e8m0fnu
    )


# 每个 Query 仅读取自己的新 KV；-1 屏蔽其余索引槽位。
window_cache_slots = torch.arange(P, P + T, dtype=torch.int32, device=device)
window_indices = torch.full((T, P), -1, dtype=torch.int32, device=device)
window_indices[:, 0] = window_cache_slots

inputs = dict(
    hidden_states=torch.ones((T, 5120), dtype=torch.bfloat16, device=device),
    query_norm_weight=torch.ones((1, 1280), device=device),
    kv_norm_weight=torch.ones((1, 512), device=device),
    query_down_proj_weight=wa,
    kv_proj_weight=wkv,
    query_up_proj_weight=wb,
    query_down_proj_scale=unit_mx_scale((1280, 80, 2)),
    kv_proj_scale=unit_mx_scale((512, 80, 2)),
    query_up_proj_scale=unit_mx_scale((H * 512, 20, 2)),
    k_cache=torch.zeros((2, P, 1, 544), dtype=torch.uint8, device=device),
    extra_k_cache=torch.zeros((4, P, 1, 320), dtype=torch.uint8, device=device),
    indices_in_kvcache=window_indices,
    extra_indices_in_kvcache=torch.full((T, 4 * P), -1, dtype=torch.int32, device=device),
    attn_sink=torch.zeros((1, H), device=device),
    rope_cos=torch.ones((T, 32), device=device),
    rope_sin=torch.zeros((T, 32), device=device),
    window_cache_slots=window_cache_slots,
)
out = torch.empty((T, H, 512), dtype=torch.bfloat16, device=device)
result = mega_mla(**inputs, out=out)
torch.npu.synchronize()
assert result is out
torch.testing.assert_close(result, torch.ones_like(result), rtol=1e-2, atol=2e-3)
print(tuple(result.shape), result.dtype, result.float().mean().item())
# (8, 8, 512), torch.bfloat16, 1.0
```

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python3 mega_mla_example.py
```

## 实现说明

### 代码结构

| 文件 | 职责 |
| :--- | :--- |
| [`mega_mla.py`](mega_mla.py) | 公共函数、输入准备、按线程/设备/流管理 workspace 与编译入口 |
| [`mega_mla_kernel.py`](mega_mla_kernel.py) | 设备入口及投影、cache 更新、gather、attention 的阶段调度 |
| [`attention_pipeline.py`](attention_pipeline.py) | 小 K 路径的 QK、在线 Softmax、PV 引擎及环形缓冲流水 |
| [`attention_matmul.py`](attention_matmul.py) | BF16 QK、PV 及共用的 Cube pipeline buffer 分配 |
| [`cache_decode.py`](cache_decode.py) | window FP8 和压缩 FP4 cache 的反量化 |
| [`rotary_position_embedding.py`](rotary_position_embedding.py) | 普通系数与紧凑系数的 RoPE 实现 |
| [`attention_geometry.py`](attention_geometry.py) | 几何约束、head 切分及压缩 cache 解码策略选择 |

调用关系为 `mega_mla()` → `MegaMlaOp` → `MegaMlaKernel`。主机层负责校验、
workspace 和编译缓存；设备层负责阶段分核与同步；在线 attention 的计算引擎和
buffer 生命周期集中在 `attention_pipeline.py`。这两个设备模块随 Native 包一并安装，
调用方仍只使用公共入口。

### KV cache 格式与数值计算

每条记录按“量化数据 + BF16 scale”存储，K/V 共用同一记录：

| Cache | 量化数据 | scale | 总大小 |
| :--- | :--- | :--- | ---: |
| window | 512 个 E4M3 值，512 B | 每 32 个值一组，共 16 个 BF16 值 | 544 B |
| 压缩 KV | 512 个 E2M1 值，打包为 256 B | 每 16 个值一组，共 32 个 BF16 值 | 320 B |

cache scale 按 E8M0 规则选择后，以数值 BF16 存入记录尾部。FP4 的低/高半字节分别
对应相邻偶/奇元素。该 cache scale 格式与投影权重的 E8M0 dtype 不同。

RoPE 只旋转 Q/KV 的最后 64 维，前 448 维保持原值。投影结果先舍入 BF16，再以
FP32 执行 RMSNorm，epsilon 为 `1e-20`。Q 经 RoPE 后乘 `1/sqrt(512)`；
RoPE 和缩放结果分别舍入 BF16。

### 流水与内存复用

Cube 执行投影、QK 和 PV；Vector 执行归一化、RoPE、KV 编解码、索引处理及 Softmax。
投影与压缩 KV 解码在满足调度条件时重叠；小 token 按实际核数分配工作。
H64 的部分配置使用 QK/PV 共享 L1 KV 的 Cube/Vector 流水。

| `cmp_decode_strategy` | 行为 |
| :--- | :--- |
| `predecode` | 先全量解码压缩 cache，再按索引选取 KV |
| `indexed` | 只读取并解码索引选中的压缩记录 |
| `auto` | 根据 token/head 数、索引宽度、cache 容量和核数选择 |

具体条件见 [`attention_geometry.py`](attention_geometry.py)。
两条路径使用相同的输入和 cache 格式。

函数按线程、设备、stream 和张量规格缓存内部 launcher，各 stream 使用独立 workspace；
同一线程和设备内复用编译产物。索引、页表、长度和业务数据每次重新读取。
权重准备、首次编译和 workspace 分配应放在计时循环外，图捕获前先预热相同规格。
需要释放当前线程的函数缓存时可调用 `mega_mla.clear_caches()`；该函数等待相关
stream 完成。清理前应先销毁引用这些 workspace 的已捕获计算图。

### 跨层 Query 元数据

设置 `return_sequence_metadata=True` 可导出本推理步的 Query 元数据。后续层通过
`sequence_metadata=metadata` 复用，替代 `cu_seqlens_qo/seqused_q`；两种输入不能同时提供。
每层仍须传入各自的 KV 长度、索引、页表、TopK 长度及写入位置。

元数据拥有独立存储。进入下一推理步、请求重排或 Query 长度变化后，应重新生成；
跨 stream 使用时由调用方建立依赖。

## 精度验证

测试使用独立 PyTorch 参考，覆盖投影和 MX scale、cache 编解码、分页与序列长度、
无效索引、元数据复用、不同 head/token 配置以及重复调用。测试中保留 cache/gather
逐字节比对及投影尾块边界检查。

从仓库根目录执行：

```bash
python3 -m pytest -q test/attention/mega_mla -m 'not npu and not ascendc_toolchain'
python3 -m pytest -q test/attention/mega_mla -m ascendc_toolchain
ASCEND_RT_VISIBLE_DEVICES=0 python3 -m pytest -q test/attention/mega_mla -m npu
```

测试依赖、覆盖范围和性能采集方式见
[测试说明](../../../test/attention/mega_mla/README.md)。

## 性能测试

现有测试入口提供普通执行和图重放的性能采集，支持在每次调用前施加缓存压力：

```bash
python3 test/attention/mega_mla/test_mega_mla.py benchmark \
  --tokens 8 --heads 16 --k 640 --warmup 10 --repeat 50 --pressure-mib 256 --graph
```

使用 msprof 采集设备耗时的命令见测试说明。比较结果时保持设备、软件版本、输入规格、
预热次数和缓存压力一致；设备计时不包含权重准备与首次编译。

### 执行与资源约束

无效 KV 使用的零行由 P0 的 AIV 在 kernel 内初始化，沿用 B0 向所有消费者发布；主机准备不执行清零 kernel。
P1 的 A、scale-A 和 atomic 输出按有效 M 行数裁剪，支持 T>64 且存在 M 尾块的情况。
批量 Softmax 按当前 UB 容量扣除 16 KiB 余量选择行批量；余量用于分配对齐、同步 scratch 和编译器临时资源。
FP4 展开1/2/4复用同一 scale 准备 helper，数据解码顺序与各自展开宽度保持一致。

可选环境变量 QBMM_MX_L2_SIZE_BYTES 是 MXFP8 矩阵乘的 L2 绕过阈值（字节），在构造阶段读取。
默认值为134217728；缺失、非整数或非正数使用默认值，不影响模块导入。
