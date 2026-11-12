# 源码构建

本文介绍开发环境配置、算子源码获取、运行和精度测试的方法。调用算子时，CANNBot-DSL完成所需的编译并执行计算。

## 环境准备

### 软件依赖

| 软件 | 版本或要求 |
| --- | --- |
| 操作系统 | Linux，CPU架构为`x86_64`或`aarch64`。 |
| Python | 3.10、3.11、3.12或3.13。 |
| CANNBot-DSL | 0.8.2。 |
| CANN | 9.2.1，安装与目标NPU匹配的Toolkit和算子包。 |
| PyTorch、`torch_npu` | 选择与Python、CANN及目标NPU配套的版本。 |

### 安装CANN

从[CANN下载页面](https://www.hiascend.com/zh/cann/download?versionId=813&ids=d806%2Ch0501%2Ch0601%2Ch0703&currentTab=0)获取CANN 9.2.1软件包，并按[CANN软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstWizard)安装驱动、固件、Toolkit和算子包。

根据实际安装路径加载环境变量。例如，安装在`/usr/local/Ascend`时执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

检查设备是否正常：

```bash
npu-smi info
```

### 安装Python依赖

安装配套的PyTorch和`torch_npu`后，使用同一个Python环境安装DSL和样例测试依赖：

```bash
python -m pip install cannbot-dsl==0.8.2
python -m pip install numpy ml_dtypes pytest pytest-xdist==3.8.0 PyYAML
```

验证DSL版本及NPU是否可用：

```bash
python -c 'import cannbotdsl; print(cannbotdsl.__version__)'
python -c 'import torch, torch_npu; print(torch.npu.is_available())'
```

预期分别输出`0.8.2`和`True`。

## 获取源码

```bash
git clone -b master https://gitcode.com/cann/cannbot-dsl.git
cd cannbot-dsl
```

## 运行算子

以[Add样例](https://gitcode.com/cann/cannbot-dsl/blob/master/samples/vector/add/README.md)为例，在仓库根目录将样例目录加入模块搜索路径：

```bash
export PYTHONPATH="$PWD/samples/vector/add:$PYTHONPATH"
```

以下示例计算两个Tensor的逐元素加法，并与PyTorch结果比较：

```python
import torch
import torch_npu

from add import add


x = torch.arange(16, dtype=torch.float32, device="npu")
y = torch.ones_like(x)
out = add(x, y)

torch.testing.assert_close(out.cpu(), (x + y).cpu())
print("Add校验通过")
```

将上述代码保存为`run_add.py`，在同一终端运行：

```bash
python run_add.py
```

预期输出：

```text
Add校验通过
```

## 精度测试

在仓库根目录运行Add已有的精度测试：

```bash
python -m pytest test/vector/add/test_add.py -v
```

其他算子的源码、参数和测试方法见[样例导航](https://gitcode.com/cann/cannbot-dsl/tree/master/samples)。
