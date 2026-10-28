# idx2crd

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

根据参数 `shape`，将从 0 开始的线性索引 `idx` 按行主序换算为多维坐标。末维坐标最先连续变化，该维遍历完毕后再向前一维进位；结果保留 `shape` 的嵌套结构。

## 函数原型

```python
idx2crd(idx, shape: Shape) -> Coord | _Coord
```

备注：`idx` 和 `shape` 均为静态值时，返回 Python 整数或元组形式的坐标；包含运行时值时，返回内部坐标对象 `_Coord`。`_Coord` 未作为公开接口导出，由 `idx2crd()` 返回。

## 参数说明

| 参数名 | <span style="white-space: nowrap">输入/输出</span> | 说明 |
| --- | --- | --- |
| `idx` | 输入 | <p>必填。从 0 开始的线性元素索引。可以是以下类型：</p> <ul><li><code>int</code>：Python 整数索引。</li><li><code>Kernel 运行时整数值</code>：例如 Kernel 参数、循环索引或整数计算结果。</li></ul><p>取值须满足 <code>0 ≤ idx &lt; shape 各维长度的乘积</code>。</p> |
| `shape` | 输入 | <p>必填。各维坐标范围的逻辑形状。可以是以下类型：</p> <ul><li><code>int</code>：一维正整数长度。</li><li><code>tuple</code>：各维长度组成的元组，可嵌套；返回坐标保留相同的嵌套结构。</li><li><code>运行时整数值或含运行时整数值的 tuple</code>：指定动态维度，参与除法和取模的维度长度须为正整数。</li></ul> |

## 返回值说明

返回 `idx` 在 `shape` 中对应的多维坐标。对于普通多维
`shape=(s0, s1, ..., sn)`，第 `i` 维坐标按以下方式计算：

```text
coord[i] = (idx // 后续各维长度的乘积) % shape[i]
```

最后一维之后没有其他维度，其“后续各维长度的乘积”按 1 计算。例如
`idx=11`、`shape=(5, 4)` 时：

```text
coord[0] = (11 // 4) % 5 = 2
coord[1] = (11 // 1) % 4 = 3
```

因此返回 `(2, 3)`。

返回坐标的结构与 `shape` 一致。例如：

| `shape` | `idx` | 返回值 |
| --- | --- | --- |
| `8` | `3` | `3` |
| `(5, 4)` | `11` | `(2, 3)` |
| `((2, 3), 4)` | `11` | `((0, 2), 3)` |

当 `idx` 和 `shape` 都由 Python `int` 组成时，返回 Python `int` 或 tuple，并在
编译阶段完成计算。只要其中包含 Kernel 内的整数值，就返回可在 Kernel 中继续用于
索引、计算或拆分的坐标值。

## 约束说明

无

## 调用示例

```python
import cannbotdsl as cbd

print(cbd.idx2crd(11, (5, 4)))
print(cbd.idx2crd(11, ((2, 3), 4)))
```

输入：

```text
idx=11
shape=(5, 4)、((2, 3), 4)
```

输出：

```text
(2, 3)
((0, 2), 3)
```
