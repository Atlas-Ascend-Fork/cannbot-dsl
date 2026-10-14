# 运行测试

测试需要在满足项目依赖且具有对应 Ascend NPU 环境的服务器上运行。

## 安装依赖

在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
```

## 运行单个样例测试

例如运行 MatMul 测试：

```bash
python -m pytest test/matmul/matmul/test_matmul.py -v
```

修改一个样例时，优先运行与它直接对应的测试。提交前再根据仓库 CI 要求扩大验证范围。
