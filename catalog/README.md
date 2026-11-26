# CI 算子清单

[operators.json](operators.json) 定义 CI 使用的算子 ID、源码路径、测试路径和测试路由。

| 字段 | 含义 |
| --- | --- |
| `operators` | 以算子 ID 为键的定义；`source`、`test` 使用仓库根目录相对路径 |
| `ci.inventory` | CI 的源码覆盖范围及顺序 |
| `ci.rules` | 规则匹配顺序、启用状态及联动测试 |
| `inventory_path`、`ci_test` | 源码覆盖范围或测试目录与算子目录不一致时的路径配置 |

`@<ID>/source/<文件>`、`@<ID>/test` 引用已登记的源码或测试路径。

修改 CI 相关清单后，在仓库根目录生成并检查配置：

```bash
python3 scripts/ci/sync_catalog.py --write
python3 scripts/ci/sync_catalog.py --check
```

生成器使用 Python 标准库，输出以下文件：

- [CI 算子清单](../scripts/ci/operator_list.yaml)
- [CI 测试路由](../test/test_config.yaml)

CI 清单与生成文件需一并提交。

`scripts/ci/sync_catalog.py` 直接读取清单、展开路径引用并校验配置。
该目录仅供 CI 使用，不参与 Native 打包。

网络配置对应的 CI 规则为 `network_groups`，保持 `enabled: false`。
修改该策略时需同时更新生成的 CI 配置。
