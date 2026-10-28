# 源码构建

本文介绍 CANNBot-DSL 源码的环境准备、源码下载、编译安装和测试流程。章节结构与 HCCL 源码构建文档保持一致。本文只介绍宿主机部署方式；本项目当前不提供 Docker 构建方式。

## 环境准备

本项目支持源码构建。编译前请完成基础环境搭建、安装 CANN 开发套件并准备仓库源码。NPU 驱动和固件属于运行态依赖，仅编译源码时可以不安装；运行样例或执行 NPU 测试前必须完成安装。

### 前置依赖

本项目编译和测试用到以下软件，请注意版本要求：

- Linux，CPU 架构为 `x86_64` 或 `aarch64`；
- Python 3.10、3.11 或 3.12；
- CANNBot-DSL 0.7.x；
- CANN Toolkit 开发套件包；
- `pip` 及仓库 `requirements.txt` 中列出的 Python 依赖；
- 与目标 NPU 匹配的 PyTorch、`torch_npu`（运行样例和 NPU 测试时需要）。

### 安装CANN软件包

#### 1. 安装驱动与固件（运行态依赖）

驱动与固件的下载和安装操作请参考《[CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)》中“准备软件包”和“安装NPU驱动和固件”章节。驱动与固件是运行态依赖，若仅编译本项目源码，可以不安装。

#### 2. 安装CANN软件包（开发态依赖）

编译本项目依赖 CANN Toolkit 开发套件包。请根据操作系统架构和目标 NPU 产品下载匹配的 Toolkit 安装包，并参考《CANN软件安装指南》完成安装。运行样例或执行 NPU 测试时，还需要安装匹配的 CANN 算子包。

### 环境验证

安装 CANN 软件包后，需验证环境是否正常：

```bash
npu-smi info
cat /usr/local/Ascend/cann/*-linux/ascend_toolkit_install.info
```

### 环境变量配置

根据实际安装路径加载 CANN 环境变量。root 用户默认安装路径示例：

```bash
source /usr/local/Ascend/cann/set_env.sh
```

非 root 用户或自定义安装路径请替换为实际的 `${install_path}/cann/set_env.sh`。

## 编译安装

### 下载源码

```bash
git clone -b master https://gitcode.com/cann/cannbot-dsl.git
cd cannbot-dsl
```

### 编译

本项目提供统一的 Native 算子包构建入口。进入仓库根目录后，可先查看可构建的算子和算子组：

```bash
cd net/native_package
./run-build.sh --list
```

构建全部已登记内容：

```bash
./run-build.sh
```

也可以指定算子组、算子和目标架构：

```bash
./run-build.sh --group ds41
./run-build.sh --operator mqsmla
./run-build.sh --group ds41 --target dav-2201
```

默认目标架构为 `dav-3510`。构建脚本会根据 `operator_groups.toml` 查找 `samples/` 中登记的源码并生成安装包。

### 安装

构建成功后，安装输出目录中的 wheel：

```bash
python3 -m pip install --force-reinstall output/groups/ds41/wheels/*.whl
```

实际路径以构建日志和 `net/native_package/output/` 中的产物为准。

### 卸载

卸载构建安装的 wheel 时，请使用实际安装包名称：

```bash
python3 -m pip uninstall <package-name>
```

## 测试

### 单元测试

```bash
python3 -m pytest net/native_package/test_build.py -v
```

### 样例精度测试

```bash
python3 -m pytest test/matmul/matmul/test_matmul.py -v
```

### NPU测试

运行 NPU 测试前，请确认驱动、固件、CANN 软件包、PyTorch 和 `torch_npu` 已正确安装并完成环境变量配置：

```bash
bash build.sh --mode=npu
```

没有 NPU 环境时可运行 CPU 侧检查：

```bash
bash build.sh --mode=cpu
```

## 附录

### 构建产物

构建缓存和产物位于 `net/native_package/`：

```text
net/native_package/
├── .build/   # 编译缓存和临时文件
└── output/   # Native、AICPU 和 wheel 产物
```

### 常用构建参数

- `--list`：列出已登记的算子和算子组；
- `--group <name>`：构建指定算子组；
- `--operator <name>`：构建指定算子；
- `--target <arch>`：指定目标 NPU 架构。
