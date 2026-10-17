# scatter_elements

## 产品支持情况

- Ascend 950：支持。

## 功能说明

`scatter_elements` 根据 `indices` 沿指定 `axis` 把 `updates` 写入 `data` 的副本。实现位于 [`scatter_elements.py`](scatter_elements.py)。对于 `indices` 的位置 `p`，目标位置由 `p` 的非索引轴坐标与 `indices[p]` 的索引轴坐标组成。`reduction="none"` 替换目标值，`"add"` 和 `"mul"` 分别在原值上逐次累加、累乘。输出 shape、dtype 和设备与 `data` 相同，输入不被原地修改，仅支持前向计算。

索引须满足 `0 <= indices[p] < data.shape[axis]`；负索引不做归一化。设备内核会跳过越界值以避免非法访存，但越界值不属于有效输入。替换操作通过整数视图保留原始位模式，包括 NaN payload、负零和 complex 值。

## 函数原型

```python
scatter_elements(
    data: torch.Tensor,
    indices: torch.Tensor,
    updates: torch.Tensor,
    axis: int,
    *,
    reduction: str = "none",
    core_num: int | None = None,
    deterministic: bool = False,
) -> torch.Tensor
```

三个输入必须连续、位于同一 NPU 且 rank 相同，rank 范围为 1～8；`data` 与 `updates` 的 dtype 相同，二者均不允许 `requires_grad=True`。`indices` 支持 int32/int64。每维须满足 `indices.shape[d] <= updates.shape[d]`，非索引轴还须满足 `indices.shape[d] <= data.shape[d]`。`axis` 范围为 `[-data.ndim, data.ndim)`，`core_num` 为可选 Vector Core 数上限。

`none` 支持 float16、bfloat16、float32、float64、int8、uint8、int16、int32、int64、bool 和 complex64；`add`、`mul` 支持其中除 float64、complex64 外的类型。空索引返回 `data` 的副本。

`deterministic=True` 时，重复目标按 `indices` 的展平顺序处理，替换操作最后一次写入生效。默认非确定性替换不保证重复目标的写入顺序，结果应来自该目标的某次合法更新。float32 原子加法的求和顺序也可能与 CPU 顺序参考不同。

## 测试

配置 CANN 环境并进入已安装 PyTorch、torch_npu、cannbotdsl 和 pytest 的 Python 环境，从仓库根目录执行：

```bash
python -m pytest -q test/scatter_elements/test_scatter_elements.py
```

默认执行 10 条 NPU 用例和 13 条主机校验。NPU 用例包含原有 5 条指定规格的功能与性能用例，以及原始位模式保留、int64 高位加法与乘法、float16 有序乘法、axis=0 的 float32 确定性加法 5 条典型回归。NPU 不可用时跳过设备测试。仅运行指定规格用例或主机侧测试可执行：

```bash
python -m pytest -q test/scatter_elements/test_scatter_elements.py -k test_functional_precision_case
python -m pytest -q test/scatter_elements/test_scatter_elements.py -m "not npu"
```

输入和 CPU 参考结果通过固定随机种子在内存中生成，不保存输入或参考数组。指定规格用例在功能校验通过后，预热 1 次、采集 2 次设备耗时，结果默认写入 pytest 临时目录；可用 `CANNBOT_CASE_RESULTS` 指定本次运行的输出目录。直接运行测试脚本同样执行全部测试，并额外汇总 `performance.csv`。

## 性能

以下为 2026-10-14 在 Ascend 950DT、Python 3.12 和同一 CANN 环境下配对采集的结果。Ascend C 与 DSL 使用相同输入；时间为设备侧 Task Duration 的最小值，不包含编译、主机调用和输入搬运耗时。Ascend C kernel 模式采集 31 个样本，ACLNN 模式采集 30 个样本；DSL 每例采集 30 个样本。所有 DSL 用例通过独立 CPU 参考校验。

`dsl/ascendc = DSL 时间 / Ascend C 时间`，小于 1 表示 DSL 更快。

| 用例 | Ascend C (μs) | DSL (μs) | dsl/ascendc |
|:---|--:|--:|--:|
| `scatter_elements_determ_none_008` | 3.980 | 2.780 | 0.6985 |
| `scatter_elements_none_027` | 5.662 | 4.474 | 0.7902 |
| `scatter_elements_add_014` | 4.580 | 3.544 | 0.7738 |
| `aclnnScatter_fp16_int64_nd_infnan_random_000016` | 5.276 | 4.052 | 0.7680 |
| `ScatterElements_int64_int32_ND_fuzz_000075` | 4.343 | 3.362 | 0.7741 |

5 条 Ascend C 基线均使用 `ops-nn` 内核，包括名称含 `aclnnScatter` 的用例。所有索引均已调整到合法范围。

此表使用配对输入及上述采样方式。测试脚本重新生成输入并使用 2 次采样，其单独测量结果不能与表中的 Ascend C 时间直接组合计算比值。
