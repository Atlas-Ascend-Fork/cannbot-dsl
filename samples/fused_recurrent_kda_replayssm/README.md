# fused_recurrent_kda_replayssm

## 产品支持情况

- Ascend 950：支持。
- 支持 8、24、28、32 AIC 配额；默认选择当前 stream 配额内最大的支持值。

## 功能说明

独立 ReplaySSM verify 算子从只读 FP32 checkpoint 计算候选 token 的 recurrent
KDA 输出，并写出后续 Commit 使用的 FP16 U/K 和 FP32 decay 记录。输入 Q/K/V
已经完成投影与卷积；算子内部完成 Q/K 归一化、门控激活和 FP32 状态递推。

每个 token 的计算为：

```text
q = Q / sqrt(sum(Q²) + 1e-6)
k = K / sqrt(sum(K²) + 1e-6)
decay = exp(lower_bound * sigmoid(exp(A_log) * (g + dt_bias)))
state = state * decay
u = sigmoid(beta) * (V - state @ k)
state = state + u ⊗ k
output = state @ q * scale
```

递推中的 `state`、`u` 和 `k` 使用 FP32；输出转换为 BF16，保存的 U/K 记录
转换为 FP16。算子不会把最终状态写回 `recurrent_state`。接受候选 token 后，
调用 `commit_recurrent_kda_replayssm` 消费这些记录，按 accepted 前缀更新 checkpoint。

## 接口说明

```python
fused_recurrent_kda_replayssm(
    mixedqkv, recurrent_state, beta, g, scale, A_log, dt_bias,
    lower_bound=-5.0, layout_qkv="BSND", *,
    replay_u, replay_k, replay_decay, cu_seqlens=None, block_num=None,
)
```

Q/K/V 头数相同，按 Q、K、V 顺序打包，`D=128`。返回值与 `g` 具有相同 shape，
数据类型为 BF16。

| 参数 | 类型 | shape / 说明 |
|:---|:---|:---|
| `mixedqkv` | BF16 | BSND：`[B,S,3*N*128]`；BNSD：`[B,3*N,S,128]`；TND：`[T,3*N*128]` |
| `recurrent_state` | FP32 | 只读 `[B,N,128,128]`，最后两维依次为 value 和 key |
| `beta` | BF16 | BSND：`[B,S,N,1]`；BNSD：`[B,N,S,1]`；TND：`[T,N,1]`，为 sigmoid 前的 logits |
| `g` | BF16 | BSND：`[B,S,N,128]`；BNSD：`[B,N,S,128]`；TND：`[T,N,128]` |
| `scale` | float 或 None | 有限值；None 使用 `128**-0.5` |
| `A_log` | FP32 | `[N]` |
| `dt_bias` | FP32 | `[N,128]` |
| `lower_bound` | float | 默认 -5.0，范围 `[-5,0]` |
| `layout_qkv` | str | BSND、BNSD 或 TND，默认 BSND |
| `replay_u`、`replay_k` | FP16 | 原地写入的独立 `[B,C,N,128]` 记录 |
| `replay_decay` | FP32 | 原地写入 `[B,C,N,128]`，保存 decay 的指数值 |
| `cu_seqlens` | INT32 | TND 必填 `[B+1]` 前缀和；其余布局无需提供 |
| `block_num` | int 或 None | 可选 8、24、28、32；显式值不得超过当前 stream 的有效配额 |

## 约束说明

- `B>0`，`1≤N≤96`，`D=128`，序列长度为正数。
- 所有 Tensor 连续，并位于同一 NPU。TND prefix sum 从 0 严格递增到 T，
  每个序列的长度不得超过记录容量 C；调用方保证 NPU 上的 prefix sum 数值合法。
- BSND/BNSD 要求 `C≥S`；TND 记录按 batch 从位置 0 保存，未使用的后缀保持原值。
- verify 计算全部候选 token，不接受 accepted 参数；Commit 单独选择接受前缀。
- 一个动态 compiled callable 复用布局、shape 与启动配额。

## 调用示例

```python
output = fused_recurrent_kda_replayssm(
    mixedqkv, checkpoint, beta, g, None, A_log, dt_bias,
    layout_qkv="BSND", replay_u=replay_u, replay_k=replay_k,
    replay_decay=replay_decay,
)
commit_recurrent_kda_replayssm(
    checkpoint, replay_u, replay_k, replay_decay, accepted_tokens,
)
```

## 精度测试

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python -m pytest -q test/fused_recurrent_kda_replayssm/test_fused_recurrent_kda_replayssm.py
```

3 个精度 case 覆盖 BSND/BNSD/TND、32/64/128 行分块、跨 wave 调度、
变长序列、不同 accepted 前缀及记录后缀保留；另有 `B=1,N=1,S=1,D=128`
的最小全量精度 case。
独立 CPU FP32 reference 检查输出、U/K/decay 记录和 Commit 后的状态，精度容差
为 `atol=rtol=0.005`；只读 checkpoint、accepted=0 和未使用的记录后缀使用零容差。
