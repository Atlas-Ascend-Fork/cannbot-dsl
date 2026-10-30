# Native 打包逻辑

本目录通过 `cannbotdsl.aot.build` 接口完成 Native 编译与 wheel 打包。算子源码直接取自仓库的 `../../samples/` 目录，**无需复制到别处**：在 `operator_groups.toml` 中登记算子名与源文件名即可。

算子名、源码文件和组定义集中维护在 `operator_groups.toml`：

```toml
[operators]
mqsmla = ["mixed_quant_sparse_flash_mla.py", "mixed_quant_sparse_flash_mla_metadata.py"]
qli = ["quant_lightning_indexer_dsl.py", "quant_lightning_indexer_metadata_dsl.py"]
qsli = ["quant_sparse_lightning_indexer_dsl.py", "quant_sparse_lightning_indexer_metadata_dsl.py"]

[groups]
ds41 = ["mqsmla", "qli", "qsli"]
```

算子名必须与源码中的 `@export(...)` 名称一致。新增算子时，在 `[operators]` 中登记其源文件名（按**文件名**在 `samples/` 下递归查找，因此文件放在 `samples/` 的任意子目录都可以）；需要加入组时，再把算子名加入对应的 `[groups]` 数组。

**算子用到的辅助模块必须一并列在它的文件数组里**（例如 `matmul_streamk = ["matmul_streamk.py", "_sample_raw_vf.py", "device_properties.py"]`）。漏列会导致打出的 wheel 内缺模块，装载后 import 失败。

采样文件的基名必须唯一：扁平化后文件名即模块名，重名会被构建脚本直接拒绝。

## 构建命令

在 `net/native_package` 下执行：

```bash
# 查看可用算子和组
./run-build.sh --list

# 编译并打包全部已登记算子
./run-build.sh

# 编译并打包一个已登记组
./run-build.sh --group ds41

# 编译并打包单个算子
./run-build.sh --operator mqsmla
# 也可以传源文件名
./run-build.sh --operator mixed_quant_sparse_flash_mla.py

```

默认目标是 `dav-3510`，可通过 `--target dav-2201` 切换。`run-build.sh` 会优先选择 `net/` 下修改时间最新的 `cannbotdsl-*.whl`。也可设置 `CANNBOTDSL_WHEEL=/path/to/cannbotdsl.whl` 指定版本；如果使用 run-branch 构建目录，可设置 `CANNBOTDSL_ROOT=/path/to/cannbot-dsl`。设置 `CANNBOTDSL_ROOT` 时从 `build/run/payload/` 查找。需要包含 `cannbotdsl.aot.export/build/register_directory` 的 0.7.0 wheel，无需独立 OpKit。

构建前请设置 `CANN_ENV=/path/to/set_env.sh`，脚本也会尝试发现常见的 `$HOME/Ascend` 安装路径。

## 目录隔离

构建按选择类型分别保存，彼此不会覆盖：

```text
native_package/
├── .build/
│   ├── frameworks/<dsl-wheel-sha256>/
│   ├── groups/ds41/
│   ├── operators/mqsmla/
│   └── all/
└── output/
    ├── groups/ds41/{native,aicpu,wheels}/
    ├── operators/mqsmla/{native,aicpu,wheels}/
    └── all/{native,aicpu,wheels}/
```

`.build/` 包含编译缓存、临时目录、Python 缓存以及被替换的上一版成功产物；`output/` 只包含当前成功发布的 Native、AICPU 目录和 wheel。构建失败时不会替换已有的成功产物。

构建脚本将 DSL wheel 安装到隔离目录，wheel 内容变化时使用新缓存。生成的算子 wheel 依赖 `cannbot-dsl>=0.8.0`，导入 `ops` 时通过 `cannbotdsl.aot.register_directory` 注册 Native 资源。

本工程打包 DS41 的 `mqsmla`、`qli`、`qsli` 和 `attn_epilogue`，其中前三个同时携带 metadata 模块。AICore 通过 `cannbotdsl.aot` 导出；AICPU metadata 在各自独立的编译目录生成，随 wheel 安装到 `ops/_aicpu/`，运行时优先加载这些预编译文件。源码直接运行时仍保留原有的 metadata 编译路径。

默认导出 MQSMLA 的 36 个配置、QLI 和 QSLI 各 18 个配置。QLI 使用 N=32、每块 6 行 query、TopK=512，可启用 2048 个候选块；QSLI 使用 2048 个候选块、TopK=512。两者覆盖 PA_BBND/TND、mask_mode=0/3、因果压缩比 1/2；PA 页大小为 64/128。QLI 覆盖候选开启/关闭，QSLI 覆盖输出偏移有/无。B、T1 和 K 序列长度为动态轴；默认 profile 的序列长度输入均存在。`attn_epilogue` 的 `T` 参与编译，默认导出 28/32 核 × `T=1,72,128,256,512,1024,2048,4096,8192` 共 18 个配置。

其他静态配置可通过 `CANNBOTDSL_DS41_PROFILES=/path/to/profiles.json` 指定，JSON 的 `qli`、`qsli` 键各对应配置字典列表，字段见相应 `export_qli`、`export_qsli`；`attn_epilogue` 键为 `{"block_num": int, "tokens": [int, ...]}` 列表，字段见 `export_attn_epilogue`。只列出的配置会被替换，未列出的算子保留默认配置。全量包指全部登记算子及其默认导出配置，不表示任意静态参数组合均有二进制。

安装后可设置 `CANNBOTDSL_NATIVE_BINARY_MODE=require` 验证二进制覆盖，未命中时立即报错，不回退 JIT。设置 `CANNBOTDSL_NATIVE_BINARY_REPORT=/path/to/native-report.json` 可保存命中记录。

`--list` / `--help` 无需 CANN 环境或 wheel。实际构建前指定 DSL wheel，例如（路径请替换为实际安装位置）：

```bash
export PYTHON=/path/to/python3.12
export CANN_ENV=/path/to/Ascend/cann/set_env.sh
export CANNBOTDSL_WHEEL=/path/to/cannbotdsl-0.8.0-cp312-cp312-manylinux_2_28_x86_64.whl
./run-build.sh --group ds41
```

wheel 分发名保留 `cannbot-arena-net`，Python 导入名仍为 `ops`。不要与同样提供 `ops` 的旧分发 `cannbot-arena-net-ops` 混装。
