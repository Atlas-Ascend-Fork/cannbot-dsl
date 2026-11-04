# 算子测试

测试按 `attention`、`matmul`、`vector`、`mc2` 分类，与
[算子样例](../samples/README.md)对应。每个目录包含算子测试、参考结果计算及参数集。

在仓库根目录执行：

```bash
python3 -m pytest test/attention/flash_kda -v
python3 -m pytest test/vector -v
bash scripts/ci/run_tests.sh --ops=attention/flash_kda --mode=cpu
```

CPU、工具链和 NPU 用例的环境要求见测试标记和对应算子 README。
`conftest.py` 提供共享配置；`_samples_path.py` 按相对 `samples/` 的路径加载样例，
例如 `attention/flash_kda/flash_kda.py`。

[test_config.yaml](test_config.yaml) 定义 CI 的测试选择规则：`src` 使用仓库根目录
相对路径，`options` 使用 `test/` 相对路径，支持选择算子目录或具体测试文件。
规则在 [算子清单](../catalog/README.md)中维护。

Moonep 权重预取的独立测试和设备性能基准位于 `mc2/moonep/`。
