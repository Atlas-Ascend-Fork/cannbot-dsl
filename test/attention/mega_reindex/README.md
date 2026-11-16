# mega_reindex 测试

测试、调用示例、迁移验证和性能分析统一位于 [`test_mega_reindex.py`](test_mega_reindex.py)，
参考 mega_mla 的单文件组织方式。接口约束见[算子说明](../../../samples/attention/mega_reindex/README.md)。
以下命令均从仓库根目录执行。

## 测试

```bash
# 查看用例和统一入口。
python3 test/attention/mega_reindex/test_mega_reindex.py --help
python3 -m pytest --collect-only -q test/attention/mega_reindex

# 主机侧契约、Native 导出与 query tiling。
python3 test/attention/mega_reindex/test_mega_reindex.py test -m 'not npu' -q

# 全部设备精度、PA、多 batch、核配额及 Graph 回归。
python3 test/attention/mega_reindex/test_mega_reindex.py test -m npu -q -s

# 单独运行设备场景。
python3 test/attention/mega_reindex/test_mega_reindex.py smoke
python3 test/attention/mega_reindex/test_mega_reindex.py verify-pa
python3 test/attention/mega_reindex/test_mega_reindex.py verify-pa-batch
```

| 测试组       | 覆盖范围                                                                     |
| ------------ | ---------------------------------------------------------------------------- |
| 主机契约     | 公共签名、支持层、依赖、设备与 stream 资源检查、缓存命中后的重新检查         |
| Native 导出  | 默认配置、JSON 覆盖、非法输入、TensorSpec 签名、源码依赖与 ds41 登记         |
| Query tiling | 2/8/16/28/32/64 核配额及非法配额                                             |
| RMSNorm tail | 31/64/65/1279/1280 维，padding 哨兵及 CPU 参考                               |
| smoke        | 非恒等 RoPE、E8M0/UINT8 scale、重复调用、空候选、大 batch 和 Graph           |
| PA           | PA128/64/16/1024、独立页 stride、乱序页表、短 KV、无效 query 和 TND 精确比较 |
| PA workspace | 短容量和百万级逻辑容量、紧凑分配、显式或自动 workspace、容量及元数据校验     |
| 多 batch PA  | B=2/S=6、独立 batch 页表、28/8/2/32 核及 Graph                               |

纯主机契约和 Native 导出检查只需 pytest；query tiling 还需要 CANNBotDSL、
PyTorch 和 torch_npu，缺少依赖时跳过。设备测试需要已加载 CANN 环境的 Ascend 950，
无可用 NPU 或 stream 配额 API 时跳过。三个设备场景在独立子进程中执行，分别报告结果，
每个场景保留 900 秒超时，失败或超时输出完整子进程日志。

`_make_case()` 共享确定性输入和工作区准备，PA 与 smoke 直接调用该函数，
不再截取其他测试脚本的源码。`run_example(tensors, ...)` 保留输入字典调用示例。

## 迁移与性能

```bash
# 先在原实现环境生成参考快照，setup 只负责准备输入和 program/args/workspace。
python3 test/attention/mega_reindex/test_mega_reindex.py capture /path/to/reference_setup.py /path/to/baseline.pt

# 待测环境使用内置的确定性输入准备函数。
python3 test/attention/mega_reindex/test_mega_reindex.py capture test/attention/mega_reindex/test_mega_reindex.py /path/to/candidate.pt
python3 test/attention/mega_reindex/test_mega_reindex.py compare /path/to/baseline.pt /path/to/candidate.pt
python3 test/attention/mega_reindex/test_mega_reindex.py benchmark test/attention/mega_reindex/test_mega_reindex.py /path/to/profile_output --reference /path/to/baseline.pt --samples 30

# 分析 a1/b1/b2/a2 四个独立进程的 profiler 目录。
python3 test/attention/mega_reindex/test_mega_reindex.py analyze-profile /path/to/abba /path/to/summary.json
```

`capture` 和 `benchmark` 的 source 可以是本测试文件，也可以是独立 setup 脚本。
本文件通过 `_make_case()` 准备输入；外部 setup 按普通 Python 脚本完整执行，应只定义
`program`、`args`、`candidate_block_length`、`num_blocks`、`front_workspace` 和
`tail_workspace` 等输入与工作区，不应包含额外测试或性能采集。
参考快照必须由原实现独立生成；不能用待测结果替代参考。profiling 前后检查 score 和
indices 精确一致，每次测量前执行既有的冷缓存驱逐工作负载。
