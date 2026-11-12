# Mega / Commit TTK 用例

pytest 与 TTK 共用五份 CSV 中的设计配置。各清单的用例 ID 互不重叠。

| 清单 | 条数 |
|---|---:|
| ttk_cases_interception.csv | 175 |
| ttk_cases_redline.csv | 11 |
| ttk_cases_generalization.csv | 968 |
| ttk_cases_stc.csv | 60 |
| ttk_cases_performance.csv | 96 |

设计配置保留 S ≤ 8 且排除 28 AIC。拦截仍覆盖非法 AIC 配额。
泛化清单沿用原「其他」清单的范围。性能清单固定 BSH、32 AIC，覆盖 S4/S8、TP4/8/16、
模型 batch 1～16，算子实际 B 为 TP × 模型 batch。
性能采集需单独配置 profiling，清单校验与精度执行均不代表性能结果。

case_id 是唯一执行编号，aliases 仅用于查找保留用例的历史编号。
configuration 是完整输入配置，coverage 标识覆盖目标，title 是设计说明。
PASS_NUMERIC 表示预期通过精度比较，PASS_CAPTURE 验证捕获及重放契约，
REJECT_ATOMIC 要求异常匹配且状态和记录保持不变。执行结果以日志为准。

## 接口覆盖

- BSH/TH 输出形状、可选 cu_seqlens（int32/int64）、正长度与分块边界，S4/S8 优化与变长分支。
- checkpoint、Conv 和三个 replay 池共用 PA 块数，各自支持独立的第 0 轴步长，并统一使用 `state_indices[b]` 寻址。
- replay shape 为 `[block_num,N,S,D]`。覆盖槽 0、末槽、稀疏映射、Conv 历史接受计数、未选块、PA 间隙和记录尾部。
- Mega 的 checkpoint 只读，Conv 与记录原位返回。
- FP32 U、FP16 K、FP32 decay；PA 块数量与记录容量边界；三种 Conv 权重形状。
- 张量 dtype/shape/连续性、ND/NZ、控制值、标量、别名、Parameter、可写梯度与图捕获契约。
- Commit 泛用 head 数、长记录、按流设置核配额、零/部分/全接受前缀及未接受 NaN 尾部。
- 旧 `max_batch` 和 Commit 的旧 `block_num` 仅出现在移除参数的拒绝用例中。
- Mega 记录交给 Commit 的端到端链路；Commit 只更新所选 checkpoint 的接受前缀。

有限设计集覆盖代表值和边界，不穷举所有正长度、张量数值和地址排列。
无法合理分配的极大 token 总数等资源上限不作为已覆盖项。配置校验不等同于设备编译、STC 或精度通过。

## 执行

仓库根目录启用可用的 NPU 环境后：

```bash
python -B -m pytest -q test/attention/mega_recurrent_kda/test_mega_recurrent_kda.py test/attention/commit_recurrent_kda/test_commit_recurrent_kda.py --basetemp=/tmp/mega_kda_pytest
```

pytest 从清单中读取标记为 `pytest_smoke` 的 20 条代表用例，Mega 入口文件收集 12 条，Commit 收集 8 条。
`-m interception`、`-m functional` 和 `MEGA_KDA_SUITE` 可进一步筛选。五份 TTK 清单合计 1310 条。
环境或核数不足时单独记录阻塞。

TTK 使用框架标准 e2e 入口、`kda_ttk_ops.py` 插件和所选 CSV：

```bash
export PYTHONPATH="$PWD/test/attention/mega_recurrent_kda:$PWD/test:$PWD/samples/attention:$PWD:$PYTHONPATH"
export MKDA_RUN_DIR=/tmp/mega_kda_ttk_run
python -B -m ttk e2e -i test/attention/mega_recurrent_kda/ttk/ttk_cases_stc.csv --plugin test/attention/mega_recurrent_kda/kda_ttk_ops.py --validate
```

实际 TTK 执行须配置 CPU third-party 服务；Verify/Commit/链路均用原 CPU FP64 golden
与用户指定原 CPU 小算子 golden 三方比较，严格 `cross_check/L2` 不变；不换回独立 FP32 peer。
全量精度固定 `--seed 3 --warmup=false --task-prof=false`，不启用性能任务。
pytest 保留原 CPU FP32 数学参考与原容差；两种框架的判定标准不能混为一谈。

执行子集时从清单导出，不手写新参数或动态扩展：

```bash
python -B test/attention/mega_recurrent_kda/assets/inputs.py --suite interception --output /tmp/mega_kda_interception.csv
python -B test/attention/mega_recurrent_kda/assets/inputs.py --ids I-015 F-012 --output /tmp/mega_kda_selected.csv
```

`--ids` 能解析旧编号，导出其对应唯一新 ID。`MEGA_KDA_IMPL` / `COMMIT_KDA_IMPL`
可指定待验证实现，默认本仓实现；报告必须注明实际版本，不能把历史实验结果当作当前通过。
日志、结果、输入清单、dump 和导出子集均写到仓库外，不提交验证证据。
