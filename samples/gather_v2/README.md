# gather_v2

## 产品支持情况

- Ascend 950：支持。

## 功能说明

`gather_v2` 沿指定 `axis` 按 `indices` 取出 `x` 的元素，支持批次维、负轴和可选的负索引。实现位于 [`gather_v2.py`](gather_v2.py)。将 `axis` 和 `batch_dims` 归一化后，输出形状为 `x.shape[:axis] + indices.shape[batch_dims:] + x.shape[axis + 1:]`，输出 dtype 和设备与 `x` 相同。

设索引轴长度为 `A`。默认按 `[0, A)` 判断索引合法性；开启 `negative_index_support` 后，负索引先加一次 `A`。归一化后仍越界的位置输出零，索引轴为空时也输出零。数据通过同位宽整数视图搬运，保留 NaN payload、负零和 complex 位模式；输入不被原地修改，仅支持前向计算。

## 函数原型

```python
gather_v2(
    x: torch.Tensor,
    indices: torch.Tensor,
    axis: int,
    *,
    batch_dims: int = 0,
    negative_index_support: bool = False,
    core_num: int | None = None,
    thread_num: int | None = None,
) -> torch.Tensor
```

`x` 至少一维，支持 float16、bfloat16、float32、float64、int8、uint8、int16、uint16、int32、int64、bool 和 complex64。`indices` 支持 int32/int64，可为标量；两输入必须连续且位于同一 NPU，`x.requires_grad` 必须为 `False`。

`axis` 范围为 `[-x.ndim, x.ndim)`。`batch_dims` 允许负值，按 `indices.ndim` 归一化；归一化后须满足 `0 <= batch_dims <= min(axis, indices.ndim)`，两输入前 `batch_dims` 维的形状必须相同。`core_num` 为可选 Vector Core 数上限。`thread_num` 为可选的每核 SIMT 线程数，支持 32、64、128、256、512、1024、2048，其中 2048 按 1024 启动；默认由入口选择。

## 测试

配置 CANN 环境并进入已安装 PyTorch、torch_npu、cannbotdsl 和 pytest 的 Python 环境，从仓库根目录执行：

```bash
python -m pytest -q test/gather_v2/test_gather_v2.py
```

默认执行 15 条 NPU 用例和 5 条主机校验。NPU 用例包含原有 10 条指定规格的功能与性能用例，以及批次负索引、空 gather 轴、64 位特殊位模式、DMA 缓存复用与淘汰、短字节行尾块和极值索引 5 条典型回归。NPU 不可用时跳过设备测试。仅运行指定规格用例或主机侧测试可执行：

```bash
python -m pytest -q test/gather_v2/test_gather_v2.py -k test_functional_precision_case
python -m pytest -q test/gather_v2/test_gather_v2.py -m "not npu"
```

输入和 CPU 参考结果通过固定随机种子在内存中生成，不保存输入或参考数组。指定规格用例在功能校验通过后，预热 1 次、采集 2 次设备耗时，结果默认写入 pytest 临时目录；可用 `CANNBOT_CASE_RESULTS` 指定本次运行的输出目录。直接运行测试脚本同样执行全部测试，并额外汇总 `performance.csv`。

## 性能

以下为 2026-10-14 在 Ascend 950DT、Python 3.12 和同一 CANN 环境下配对采集的结果。Ascend C 与 DSL 使用相同输入；时间为设备侧 Task Duration 的最小值，不包含编译、主机调用和输入搬运耗时。Ascend C kernel 模式采集 31 个样本，ACLNN 模式采集 30 个样本；DSL 每例采集 30 个样本。所有 DSL 用例通过独立 CPU 参考校验。

`dsl/ascendc = DSL 时间 / Ascend C 时间`，小于 1 表示 DSL 更快。

| 用例 | Ascend C (μs) | DSL (μs) | dsl/ascendc |
|:---|--:|--:|--:|
| `gather_v2_0307` | 1.834 | 1.466 | 0.7993 |
| `gather_v2_0016` | 17.978 | 13.964 | 0.7767 |
| `gather_v2_0052` | 1.544 | 1.422 | 0.9210 |
| `gather_v2_0343` | 68.647 | 67.843 | 0.9883 |
| `gather_v2_0003` | 1.416 | 1.285 | 0.9075 |
| `gather_v2_test_case_0` | 2.660 | 2.462 | 0.9256 |
| `gather_v2_test_case_1` | 1.974 | 1.006 | 0.5096 |
| `gather_v2_test_case_2` | 1.709 | 1.024 | 0.5992 |
| `gather_v2_test_case_3` | 1.992 | 1.074 | 0.5392 |
| `gather_v2_test_case_4` | 1.968 | 1.278 | 0.6494 |

前 5 条 Ascend C 基线使用 `ops-nn` 内核，后 5 条使用 CANN ACLNN 接口；ACLNN 数值仍为设备算子耗时。越界索引按本算子的规则输出零。

此表使用配对输入及上述采样方式。测试脚本重新生成输入并使用 2 次采样，其单独测量结果不能与表中的 Ascend C 时间直接组合计算比值。
