# flash_kda_fwd TTK 测试

使用 [ops-test-kit](https://gitcode.com/cann/ops-test-kit) 的原生 E2E CSV 和 TestSpec，验证真实 `flash_kda_fwd`。软件组织参考 ops-transformer 的 [QuantBlockSparseAttn 测试](https://gitcode.com/cann/ops-transformer/tree/master/experimental/attention/quant_block_sparse_attn/tests)，CSV 按测试设计章节组织。

CPU 参考计算复用了本仓库[原生前向测试](../test_flash_kda_fwd.py)中的分块、低精度舍入和三阶段递推函数。`golden.py` 将这些计算适配到公开接口，并通过 `FlashKdaFwdTestSpec` 向 TTK 提供预期输出。版权声明保留在源码中，许可证见[仓库根目录](../../../../LICENSE)。

## 文件组织

| 文件 | 职责 |
|---|---|
| `../assets/golden.py` | 两套 CPU 数学参考及 `FlashKdaFwdTestSpec` 注册 |
| `../assets/inputs.py` | 输入生成、显式参数变更及独立 ABI 4 测试数据 |
| `../assets/compare.py` | 数值比较、固定 16 项返回协议及有效区提取 |
| `../kda_ttk_ops.py` | 真实算子调用、metadata 准备、训推上下文及异常边界检查 |

| CSV | 设计章节 | 条目数 |
|---|---|---:|
| `ttk_cases_stc.csv` | 功能，包括 Meta 输出协议及 F31 的 NPU/Meta stride 对照 | 152 |
| `ttk_cases_invalid.csv` | 异常拦截、合法反例与恢复 | 290 |
| `ttk_cases_generalized.csv` | 泛化 | 70 |
| `ttk_cases_profiling.csv` | 性能 | 25 |

共 537 条。10 条长度内容用例同时检查 metadata 生产和前向消费；常规回归为 512 条，性能单独运行。有效核数等于 ABI 的全部 32 个核槽时，两条“有效核数以外存在任务”场景不适用，不能伪造核数来执行。metadata 使用标量接口：`cu_seqlens、seqused、*、batch_size、nheads_qk、nheads_v、seqlens、headdim_qk、headdim_v、enable_grad、layout`。TND 必须提供 cu_seqlens，seqlens 在三种布局下都必须显式提供。

## 参数与检查

张量顺序与被测函数保持一致：`q, k, v, g, beta, initial_state, A_log, dt_bias, cu_seqlens, seqused`。公开属性沿用 `layout`、`lower_bound`、`return_intermediate_states` 。三个融合开关默认 True，输入为原始 q/k、g 和 beta logits，A_log 必传。lower_bound 默认 None，使用 softplus，有效数值下界使用有界 sigmoid。metadata 在测试的实际训推上下文中生成，模式由 `torch.is_inference_mode_enabled()` 决定，`no_grad()` 单独验证。

CSV 的 shape、dtype、算子属性直接进入 TTK。`valid_lengths`、`initial_state_mode`、`dt_bias_mode`、`seed` 等属性控制输入；`omitted_attributes` 指定需要从真实调用中省略的可选参数。`test_config` 仅描述额外检查，是 JSON 字符串属性，包含变更对象、方式、值、错误信息或对照操作。普通精度用例不需要该属性。用例名称仅用于结果和日志，不用于选择测试行为；运行时不回读 CSV 或按编号查表。

例如，非法 q 类型通过 `{"kind":"negative","mutation":["dtype","q","float32"]}` 描述；metadata 头损坏通过 `{"kind":"metadata","schedule":"single_round","change":"header","field":"magic","value":0,"error_patterns":["magic"]}` 描述。内部缓冲区检查按名称拦截真实 `validate_buffer` 调用，非法缓冲区必须在获取编译函数前被拒绝。stride 和合法视图直接修改实际 Tensor；合法视图另通过 `FlashKDAFwd.run` 签名核对传入内核的具名参数及 storage_offset，不固定参数位置索引。故障注入仅用于设计中的白盒场景，必须验证注入点被实际执行、异常原因匹配且撤销后恢复。

- GQA：`1 <= Nqk <= Nv <= 96` 且 `Nv % Nqk == 0`。96/97 为实现上限检查；ABI 的 `workspace_slots=861` 不是合法 head 上限。
- `TestSpec.tolerance` 声明官方 `cross_check`、`level="L2"`；`golden` 返回 `cpu_chunk`，`third_party["torch"]` 返回 `npu_chunk` CPU 模拟基准。主输出保留实际 BF16/FP32 dtype，由 TTK 选择对应阈值并记录 `precision_metrics`，不使用自定义 `compare` 钩子覆盖框架结果。MARE/MERE/RMSE 比值上限为 2.0/1.2/1.2，小值域规则也由框架解析。两套参考均在 CPU 执行，未接入独立 GPU 基准。
- 主调用直接进入 TTK 原生 L2 比较；功能对照、分段状态衔接和异常恢复中的额外调用使用相同官方比较器，逐槽指标、dtype、基准和失败原因记录为 `l2_reference_check`。主输出、末态、chunk 状态不再使用 `5e-3 allclose` 验收。有限值与详设专项容差独立记录为 `numerical_contract_check`：q/k norm 为 `atol=rtol=1e-3`，rstd/Beta 激活为 `1e-6`。空返回槽标记为 `NOT_APPLICABLE`，不计作数值验收。
- 正例按位置传入六个必填 Tensor，第六项为 `metadata`。原始固定 16 项返回检查 Tensor/None、shape、dtype；五个训练 token 缓存跟随输入布局，BNSD 按 `[B,Nv,S,D]`、BSND 按 `[B,S,Nv,D]`、TND 按 `[T,Nv,D]` 验证；仅比较有效 token、chunk 和已定义缓存。训推主结果及输入不变性使用精确比较。
- 异常用例逐条核对异常类型、报错对象、原因及拦截边界，再用合法输入验证恢复。漏传 Python 签名中的必填参数（如 `q`、`metadata` 或 metadata 入口的 `seqlens`）应抛 `TypeError`；显式传入 `None` 的必需张量、非法 shape 或取值按接口约定抛 `ValueError`。
- 真实设备路径阻塞读取 metadata 和给定的 cu_seqlens/seqused。非法长度必须在主计算缓冲区分配前抛出 ValueError，恢复合法长度后再检查数值。双设备异常需要两张 NPU。

精度用例向 TTK 返回九个有效区输出（对应原始槽 `0,1,2,11,12,13,14,15,10`），另附一个 bool 功能契约结果。前九项按实际浮点 dtype 做 L2，第十项由 `binary_equal` 校验；数值和契约指标分开保留，契约失败不遮蔽主结果 L2。槽 10 g_cumsum 的公开 dtype 为 FP32，逐序列逐 chunk 检查有效 token 的前缀和。异常/Meta 等契约用例仅在所有检查完成后返回 bool 成功标记；恢复调用的 L2 指标查看对应事件记录。bool 通过不计作数值 L2 通过。各用例按 TestSpec 的配置执行 L2 比较，并记录逐槽指标。

## 环境与执行

需要 Linux、匹配的 PyTorch/torch_npu/CANN/cannbotdsl 和 NPU。已验证 TTK 提交为 `be8a90e372738a9c785fdcb9d4bfc2c89e6f23a1`。算子源码应具备上述固定 16 项返回、开关、ABI 4 及详设定义的标量 metadata 接口。

CPU 对照通过 TTK 原生 `xpu-server` 执行。服务端使用同一版本的测试目录，将 `test/attention/flash_kda_fwd` 加入其 `PYTHONPATH`，使 `kda_ttk_ops` 及三个资产文件可加载；`CpuSimulation` 内部显式在 CPU 计算。worker 配置放在特性开发目录，例如：

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
export PYTHONPATH="$KDA_SOURCE_ROOT/test/attention/flash_kda_fwd:${PYTHONPATH:-}"
export KDA_EVIDENCE_DIR=/path/to/feature-work/ttk/evidence
cd "$KDA_SOURCE_ROOT/test/attention/flash_kda_fwd/ttk"

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

性能用例在 CSV 中显式配置 10 次预热、50 次采样。metadata 在 TTK 排除的预热阶段生成；AICore 中位数、最小值从 TTK profiler 记录汇总，metadata、Host 校验、调用及同步耗时分别记录。`KDA_EVIDENCE_DIR` 未设置时检查信息写入日志。结果、dump、验证报告及开发脚本均保存在特性开发目录，不提交到仓库。
