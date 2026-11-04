# 算子清单

[operators.json](operators.json) 定义算子 ID、源码路径、测试路径及 Native 打包依赖。
[网络清单](../net/README.md)通过算子 ID 选择依赖。

| 字段 | 含义 |
| --- | --- |
| `operators` | 以算子 ID 为键的定义；`source`、`test` 使用仓库根目录相对路径 |
| `native.export` | Native 导出名 |
| `native.sources` | Native 主模块及辅助模块路径 |
| `native_exports` | 全量 Native 构建的算子及顺序 |
| `ci.inventory` | CI 的源码覆盖范围及顺序 |
| `ci.rules` | 规则匹配顺序、启用状态及联动测试 |
| `inventory_path`、`ci_test` | 源码覆盖范围或测试目录与算子目录不一致时的路径配置 |

`@<ID>/source/<文件>`、`@<ID>/test` 引用已登记的源码或测试路径。

修改清单后，在仓库根目录生成并检查配置：

```bash
python3 scripts/ci/sync_catalog.py --write
python3 scripts/ci/sync_catalog.py --check
```

生成器使用 Python 标准库，输出以下文件：

- [CI 算子清单](../scripts/ci/operator_list.yaml)
- [CI 测试路由](../test/test_config.yaml)
- [Native 源码及分组](../packaging/native/operator_groups.toml)

清单与生成文件需一并提交。
