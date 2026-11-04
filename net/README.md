# 网络算子分组

每个网络的 `network.json` 通过 [算子 ID](../catalog/README.md)选择 Native 构建依赖。
算子实现位于 `samples/`，构建入口见 [Native 打包](../packaging/native/README.md)。

| 网络分组 | 依赖清单 | 构建说明 |
| --- | --- | --- |
| DS41 | [ds41/network.json](ds41/network.json) | [DS41](ds41/README.md) |
| K3 | [k3/network.json](k3/network.json) | [K3](k3/README.md) |

修改分组后，在仓库根目录执行 `python3 scripts/ci/sync_catalog.py --write`。
