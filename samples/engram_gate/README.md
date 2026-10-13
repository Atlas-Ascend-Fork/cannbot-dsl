# EngramGate

`engram_gate` 使用 CANNBot-DSL 融合 Engram residual gate 的双路 RMS、加权点积、
signed-sqrt sigmoid 和 residual update。

## 接口

```python
y = engram_gate(
    x,
    key,
    value,
    weight,
    image_mask=None,
    eps=1e-6,
    clamp_value=1e-6,
)
```

| 参数 | dtype | shape |
| :--- | :--- | :--- |
| `x`, `key` | BF16 | `[..., hc_mult, dim]`，ndim ≥ 2；前导维折叠为 token，2-D 表示 `hc_mult=1` |
| `value` | BF16 | `[..., dim]`（与 `x` 前导维一致） |
| `weight` | FP32 | `[hc_mult, dim]` |
| `image_mask` | BOOL，可选 | `[...]`（与 `x` 前导维一致），`True` 的 token 原样返回 `x` |
| `y` | BF16 | 与 `x` 同 shape |

`dim` 支持任意正整数：不足 64 的尾部按掩码处理；超过单行驻留 UB 预算
（`ceil(dim/64)*64 > 8448`）时自动切换为列分块流式路径（两遍扫描、串行行
处理），无 dim 上限。模型默认配置为 `dim=5120, hc_mult=4`。

计算公式：

```text
rstd = rsqrt(mean(x²) + eps) * rsqrt(mean(key²) + eps)
dot  = sum(x * weight * key) * rstd / sqrt(dim)
gate = sigmoid(copysign(sqrt(max(abs(dot), clamp_value)), dot))
y    = bf16(x + gate * value)
```

运行测试：

```bash
pytest -q test/engram_gate/test_engram_gate.py
```

## 安装包调用

源码示例与测试直接加载本仓库 sample。需要部署安装包时，使用共享
[Native 构建工程](../../net/native_package/README.md)，注册名为 `engram_gate`，所属组为
`ds41`：

```bash
export PYTHON=/path/to/python3.12
export CANN_ENV=/path/to/Ascend/cann/set_env.sh
source "$CANN_ENV"
export CANNBOTDSL_WHEEL=/path/to/cannbotdsl-0.7.0-cp312-cp312-manylinux_2_28_x86_64.whl
bash net/native_package/run-build.sh --operator engram_gate
"$PYTHON" -m pip install --no-deps --force-reinstall \
  net/native_package/output/operators/engram_gate/wheels/*.whl
```

`T`、mask ABI 与核配额参与编译（张量形状、launch grid 和 host launcher 均随之变化），
默认按 56/64 核配额 × `T=72` × 有/无 `image_mask` 共 4 个配置；`eps`/`clamp_value`
默认导出 `1e-6`。未被枚举的取值会回退 JIT。可通过
`CANNBOTDSL_DS41_PROFILES` 的 `engram_gate` 键覆盖默认集合：

```json
{"engram_gate": [{"max_blocks": 56, "tokens": [72, 256, 1024], "has_mask": [true, false]}]}
```

安装后可设置 `CANNBOTDSL_NATIVE_BINARY_MODE=require` 验证二进制覆盖是否完整，
未命中时立即报错；`CANNBOTDSL_NATIVE_BINARY_REPORT=/path/to/native-report.json`
可保存命中记录。
