# flash_kda TTK 测试

使用 [ops-test-kit](https://gitcode.com/cann/ops-test-kit) 的原生 E2E CSV 和 TestSpec，每条精度用例分别调用真实 `flash_kda` 和同一统一实现中的 `flash_kda_fwd`。软件组织参考 ops-transformer 的 [QuantBlockSparseAttn 测试](https://gitcode.com/cann/ops-transformer/tree/master/experimental/attention/quant_block_sparse_attn/tests)，CSV 按测试设计章节组织。

CPU 参考计算复用了本仓库[原生前向测试](../test_flash_kda.py)中的分块、低精度舍入和三阶段递推函数。`golden.py` 将这些计算适配到公开接口，并通过 `FlashKdaTestSpec` 向 TTK 提供预期输出。版权声明保留在源码中，许可证见[仓库根目录](../../../../LICENSE)。

两套 CPU 参考保留原有分块计算、低精度舍入和状态递推公式，以及完整 16 项参考结果。`flash_kda` 始终使用前三项参考；`flash_kda_fwd` 在推理模式使用前三项，在训练模式使用全部 16 项。训练缓存的 qg 参考保留原有定义：`cpu_chunk` 使用 FP32 归一化 Q 和乘法；`npu_chunk` 使用转存 BF16 的归一化 Q，乘 FP32 衰减后再转存 BF16。当前 L2 检查是相对 CPU 模拟标杆的误差检查，不是独立 GPU 标杆验收。

有界 gate（`use_gate_in_kernel=True` 且 `lower_bound` 非 None）的低精度参考沿用原生 `_reference_lower_matmul()` 的每 16 行公共参考点，包括对角块；有效下界 `[-5, 0)` 将块内正指数限制在 75 以内。无界 gate 和预处理 g 保留逐行因果差分，避免未使用的上三角指数溢出后乘零产生 NaN。两条路径在实数运算下等价，但 BF16 乘法操作数的舍入不同，不能将安全逐行路径无条件替换原有有界 gate 标杆。FP32 数学参考继续使用原来的安全计算。

## 文件组织

| 文件 | 职责 |
|---|---|
| `../assets/golden.py` | 两套 CPU 数学参考及 `FlashKdaTestSpec` 注册 |
| `../assets/inputs.py` | 输入生成、显式参数变更及独立 ABI 4 测试数据 |
| `../assets/compare.py` | 两个入口的 3/16 项返回协议、数值比较及有效区提取 |
| `../kda_ttk_ops.py` | 真实算子调用、metadata 准备、训推上下文及异常边界检查 |

| CSV | 设计章节 | 条目数 |
|---|---|---:|
| `ttk_cases_stc.csv` | 功能，包括 Meta 输出协议及 F31 的 NPU/Meta stride 对照 | 154 |
| `ttk_cases_invalid.csv` | 异常拦截、合法反例与恢复 | 267 |
| `ttk_cases_generalized.csv` | 泛化 | 70 |
| `ttk_cases_profiling.csv` | 性能 | 25 |

共 516 条，保留原用例参数及训练/推理配置；各 CSV 的 `api_name` 统一为 `kda_ttk_ops.flash_kda`。10 条长度内容用例同时检查 metadata 生产和前向消费；常规回归为 491 条，性能单独运行。有效核数等于 ABI 的全部 32 个核槽时，两条“有效核数以外存在任务”场景不适用，不能伪造核数来执行。此前确认删除的 24 条错误用例不在本清单中。本次 metadata 调用按详设使用标量接口：`cu_seqlens、seqused、*、batch_size、nheads_qk、nheads_v、seqlens、headdim_qk、headdim_v、enable_grad、layout`。TND 必须提供 cu_seqlens，seqlens 在三种布局下都必须显式提供。

## 参数与检查

张量顺序与被测函数保持一致：`q, k, v, g, beta, metadata, initial_state, A_log, dt_bias, cu_seqlens, seqused`。公开属性沿用 `layout`、`lower_bound`、`return_intermediate_states` 。三个融合开关默认 True，输入为原始 q/k、g 和 beta logits，开启 gate 融合时 A_log 必传。lower_bound 默认 None，使用 softplus，有效数值下界使用有界 sigmoid。输入插件在用例指定的训推上下文中生成 INT32 metadata 并填入 CSV 的第六项，真实前向直接消费该项，模式由 `torch.is_inference_mode_enabled()` 决定，`no_grad()` 单独验证。

CSV 的 shape、dtype、算子属性直接进入 TTK。`valid_lengths`、`initial_state_mode`、`dt_bias_mode`、`seed` 等属性控制输入；`omitted_attributes` 指定需要从真实调用中省略的可选参数。`test_config` 仅描述额外检查，是 JSON 字符串属性，包含变更对象、方式、值、错误信息或对照操作。普通精度用例不需要该属性。用例名称仅用于结果和日志，不用于选择测试行为；运行时不回读 CSV 或按编号查表。

例如，非法 q 类型通过 `{"kind":"negative","mutation":["dtype","q","float32"]}` 描述；metadata 头损坏通过 `{"kind":"metadata","schedule":"single_round","change":"header","field":"magic","value":0,"error_patterns":["magic"]}` 描述。内部缓冲区检查按名称拦截真实 `validate_buffer` 调用，非法缓冲区必须在获取编译函数前被拒绝。stride 和合法视图直接修改实际 Tensor；合法视图另通过 `FlashKDAFwd.run` 签名核对传入内核的具名参数及 storage_offset，不固定参数位置索引。故障注入仅用于设计中的白盒场景，必须验证注入点被实际执行、异常原因匹配且撤销后恢复。

- GQA：`1 <= Nqk <= Nv <= 96` 且 `Nv % Nqk == 0`。96/97 为实现上限检查；ABI 的 `workspace_slots=861` 不是合法 head 上限。
- `TestSpec.tolerance` 声明官方 `cross_check`、`level="L2"`；`golden` 返回 `cpu_chunk`，`third_party["torch"]` 返回 `npu_chunk` CPU 模拟基准。主输出保留实际 BF16/FP32 dtype，由 TTK 选择对应阈值并记录 `precision_metrics`，不使用自定义 `compare` 钩子覆盖框架结果。MARE/MERE/RMSE 比值上限为 2.0/1.2/1.2，小值域规则也由框架解析。两套参考均在 CPU 执行，未接入独立 GPU 基准。
- 主调用直接进入 TTK 原生 L2 比较；功能对照、分段状态衔接和异常恢复中的额外调用使用相同官方比较器，逐槽指标、dtype、基准和失败原因记录为 `l2_reference_check`。主输出、末态、chunk 状态不再使用 `5e-3 allclose` 验收。形状、有限值和训练缓存专项容差独立记录为 `numerical_contract_check`：q/k norm 的 `atol=rtol=1e-3`，rstd/Beta 激活为 `1e-6`。空返回槽标记为 `NOT_APPLICABLE`，不计作数值验收。
- 正例按位置传入六个必填 Tensor，第六项为 `metadata`。先严格检查 `flash_kda` 返回三项、`flash_kda_fwd` 返回十六项，再提取有效区域，禁止先切片掩盖错误返回数量。前三项依次为 `out`、`final_state`、`chunk_states`，检查 Tensor/None、shape、dtype；`out` 跟随输入布局，`final_state` 由 `output_final_state` 控制，`chunk_states` 仅在推理模式且 `return_intermediate_states=True` 时返回，训练时为 None。推理时完整前向的后十三项必须为 None；训练时按开关检查全部缓存，未启用项必须为 None。Aqk/Akk 仅比较有效行列，token 缓存和 FP32 g_cumsum 仅比较有效 token。两个入口的前三项有效结果及输入不变性使用精确比较。
- 异常用例逐条核对异常类型、报错对象、原因及拦截边界，再用合法输入验证恢复。漏传 Python 签名中的必填参数（如 `q`、`metadata` 或 metadata 入口的 `seqlens`）应抛 `TypeError`；显式传入 `None` 的必需张量、非法 shape 或取值按接口约定抛 `ValueError`。
- 真实设备路径阻塞读取 metadata 和给定的 cu_seqlens/seqused。非法长度必须在主计算缓冲区分配前抛出 ValueError，恢复合法长度后再检查数值。不校验混合设备，不包含 CPU/NPU 混放、跨 NPU 或缓冲区设备不一致用例；保留全 CPU 执行限制检查。

每条精度用例使用相同输入、模式和 metadata 分别调用两个入口。`inference=True` 时两者均对比前三项有效结果；`inference=False` 时 `flash_kda` 对比前三项、`flash_kda_fwd` 对比全部十六项中启用的输出。每次调用都检查输入未被修改，metadata 的 `enable_grad=not inference`。

两组实际结果各自使用官方 `cross_check/L2` 与 FP32、低精度 CPU 参考比较，逐入口指标以 `l2_reference_check` 记录；前三项还必须精确一致，以 `entry_consistency_checked` 记录。适配器向 TTK 返回 `flash_kda_fwd` 的有效结果（推理三槽、训练十六槽），再附一个 bool；该 bool 要求两个入口的精度、专项契约、一致性和额外关系检查全部通过。TTK 同时比较实际数值槽和 bool，任一入口失败都会使该用例失败，不拼接成十九个数值槽，也不复制 CSV 用例。异常/Meta 等契约用例仍返回独立的 bool，性能仍只计时统一入口，不将两次调用合并计时。迁移后需重新执行测试，历史结果不作为当前版本的验收结论。

scale 默认 `128 ** (-0.5)`，显式传 None 也使用默认缩放；F06-005 显式传入 `scale=0.0`，应保留零值。F01-007 使用两条长度不同的紧凑序列，省略前向入口的 layout，与显式 TND 对照。

同目录的 `test_flash_kda.py` 以统一入口验证训练和推理结果，并保留共享实现中 `flash_kda_fwd` 的 16 项兼容接口检查；这些兼容检查不加载独立的旧算子目录。内部缓冲区白盒检查继续使用真实实现的 `FlashKDAFwd.run` 名称。

## 性能用例配置

`ttk_cases_profiling.csv` 保留 25 条用例：9 条推理、16 条训练；编号、shape、布局和预热/采样次数不变。下列参数在 CSV 中显式设置，`omitted_attributes` 为空，不依赖算子默认值。

| 参数 | 推理 | 训练 |
|---|---|---|
| `inference` | True | False |
| `use_qk_l2norm_in_kernel`、`use_gate_in_kernel`、`use_beta_sigmoid_in_kernel` | True | True |
| `output_final_state` | True | False |
| `return_intermediate_states` | False | False |
| `disable_recompute` | True，模式关闭训练缓存 | True，保存训练缓存 |
| `allow_neg_eigval` | False | False |
| `lower_bound` | -5.0 | -5.0 |
| `scale` | `128 ** (-0.5)` | `128 ** (-0.5)` |
| `g`、`beta` 输入 dtype | BF16 | FP32 |
| `initial_state_mode` | zero：传入全零 FP32 张量 | zero：传入全零 FP32 张量 |
| `dt_bias_mode` | normal：标准正态 FP32 张量 | normal：标准正态 FP32 张量 |
| `seed` | 101 | 101 |

`dt_bias_mode="normal"` 使用独立 CPU `torch.Generator`，按 `seed=101` 生成 `[Nv, 128]` 的标准正态 `dt_bias`；该非空、非全零 bias 不受其他输入随机数消耗顺序影响。旧版对照应复用同一份 bias 数据（或用相同 PyTorch 环境和独立生成器复现），不要沿用旧测试自身的随机生成顺序。原有 `dt_bias_mode="random"` 的均匀分布保持不变。

性能路径每次仅调用新 `flash_kda`，由 inference mode 选择训推行为，不执行精度路径的双入口检查；`golden_api=disable`。旧训练基准也应设置 `return_intermediate_state=False`、`output_final_state=False`。比较约定接受旧版关闭末态返回后仍存在的写出差异，不要求 beta 内核读取宽度和矩阵布局转换、g_cumsum 清零/复制完全一致；主内核与完整接口的计时范围在正式测试时另行确定。本配置不代表已经完成新旧算子的性能验收。

## 环境与执行

需要 Linux、匹配的 PyTorch/torch_npu/CANN/cannbotdsl 和 NPU。原测试记录的 TTK 提交为 `be8a90e372738a9c785fdcb9d4bfc2c89e6f23a1`。算子源码应具备上述统一入口三项返回和兼容前向十六项返回、开关、ABI 4 及详设定义的标量 metadata 接口。

CPU 对照通过 TTK 原生 `xpu-server` 执行。服务端使用同一版本的测试目录，将 `test/attention/flash_kda` 加入其 `PYTHONPATH`，使 `kda_ttk_ops` 及三个资产文件可加载；`CpuSimulation` 内部显式在 CPU 计算。worker 配置放在特性开发目录，例如：

```yaml
remote:
  endpoints:
    - host: 127.0.0.1
      port: 9090
```

先按官方指南部署服务包，以 `python -m xpu_server.xpu_server --devices cpu --bind 127.0.0.1 --port 9090` 启动 CPU 服务，随后将此文件通过 `--config` 传给 TTK。服务或基准不可用应判失败，不回退到 `allclose`。精度标准由 TestSpec 按 dtype 声明，运行时不要通过 CLI 改用其他标准。

```bash
python -m pip install -e /path/to/ops-test-kit
export KDA_SOURCE_ROOT=/path/to/cannbot-dsl
export PYTHONPATH="$KDA_SOURCE_ROOT/test/attention/flash_kda:${PYTHONPATH:-}"
export KDA_EVIDENCE_DIR=/path/to/feature-work/ttk/evidence
cd "$KDA_SOURCE_ROOT/test/attention/flash_kda/ttk"

# 官方 CSV 和签名校验；检查结果中每条用例的 status。
python -m ttk e2e -i ttk_cases_stc.csv --plugin ../assets/golden.py --validate

# 单条默认调用。
python -m ttk e2e -i ttk_cases_stc.csv --plugin ../assets/golden.py \
  --config /path/to/feature-work/ttk/cpu_reference.yaml \
  -t F01-001 --pc 1 --device-whitelist 0 --run 1 --warmup false \
  -o /path/to/feature-work/ttk/smoke.csv

# 功能、异常和泛化按章节依次运行，单目录不并发执行。
for cases in ttk_cases_stc.csv ttk_cases_invalid.csv ttk_cases_generalized.csv; do
  python -m ttk e2e -i "$cases" --plugin ../assets/golden.py \
    --config /path/to/feature-work/ttk/cpu_reference.yaml \
    --pc 1 --device-whitelist 0 --run 1 --warmup false \
    --proc-timeout 360 --single-log --task-prof=false \
    -o "/path/to/feature-work/ttk/${cases%.csv}_results.csv"
done

# 性能独立执行，不生成超长序列 CPU golden。
python -m ttk e2e -i ttk_cases_profiling.csv --plugin ../assets/golden.py \
  --pc 1 --device-whitelist 0 --run 50 --warmup true \
  -o /path/to/feature-work/ttk/performance.csv
```

性能用例在 CSV 中显式配置 10 次预热、50 次采样。性能用例在输入准备阶段填入 metadata，并在 TTK 排除的预热阶段单独计时一次 metadata 生产调用；AICore 中位数、最小值从 TTK profiler 记录汇总，metadata、Host 校验、调用及同步耗时分别记录。`KDA_EVIDENCE_DIR` 未设置时检查信息写入日志。结果、dump、验证报告及开发脚本均保存在特性开发目录，不提交到仓库。
