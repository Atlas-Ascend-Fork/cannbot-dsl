---
title: DelayLineGroup
api_name: DelayLineGroup
category: types-and-views
api_group: kernel
layer: frontend
call_context: device
execution_unit: scalar
status: experimental
since: 待追溯
---

# DelayLineGroup

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 简介

`DelayLineGroup` 是 Kernel 内管理多组命名标量历史值的容器，主要用于软件流水。
每个字段保存一条标量序列；当前迭代通过 `push()` 写入值，后续流水阶段通过
`tap(lag)` 读取若干次迭代之前写入的值，从而恢复产生当前数据时对应的 tile 编号、
坐标或状态。

所有字段共享同一个推进位置。每次迭代结束时只需调用一次 `advance()`，全部字段便
同时推进到下一个位置，不需要为每个标量分别维护数组下标和循环回绕逻辑。
DelayLineGroup 提供两种读取方式：所有字段使用相同滞后量时，可以通过组级
`tap(lag)` 一次读取；不同字段使用不同滞后量时，可以通过 `dl.<field>.tap(lag)`
分别读取。

## 定义

### 构造函数

`DelayLineGroup(...)` 根据字段名创建多条共享推进位置的标量延迟线。每个字段内部
保存 `depth` 个 `int64` 值，按循环方式重复使用；`depth` 决定能够读取的最大滞后
范围。

#### 函数原型

```python
DelayLineGroup(
    depth: int,
    *fields: str,
) -> DelayLineGroup
```

#### 参数说明

| 参数 | 类型 | 必选 | 默认值 | 简要说明 |
| --- | --- | --- | --- | --- |
| `depth` | `int` | 是 | 无 | 指定每个字段保存的历史值数量。 |
| `fields` | `str` | 是 | 无 | 指定一个或多个字段名；字段名按位置传入。 |

##### `depth` 参数可选值

`depth` 必须是满足 `depth >= 2` 的 Python `int`，并且必须大于实际使用的最大
`lag`。例如，需要读取两次迭代之前的值时，最大 `lag` 为 2，因此 `depth` 至少为
3。

| 取值 | 含义 |
| --- | --- |
| `2` | 可以使用 `lag=1` 读取前一次迭代写入的值。 |
| `N`，其中 `N > 2` | 可以使用 `1 <= lag < N` 读取对应迭代之前写入的值。 |

##### `fields` 参数可选值

`fields` 至少包含一个字段名。每个字段名既是组级 `push()` 的关键字参数名，也是
对应单字段视图的属性名。例如：

```python
dl = DelayLineGroup(3, "batch", "head", "m", "n")
```

以上构造声明了 `batch`、`head`、`m` 和 `n` 四个字段，可以通过 `dl.batch`、
`dl.head`、`dl.m` 和 `dl.n` 分别访问。

#### 返回值说明

返回包含全部命名字段并共享推进位置的 `DelayLineGroup` 对象。

#### 约束说明

- 必须至少提供一个字段名。
- `depth` 必须满足 `depth >= 2`。
- 每个字段只保存整数标量。写入值统一按 `int64` 保存；其他整数类型必须能够无损
  转换为 `int64`。
- 创建后尚未写入的槽位没有有效历史值。读取 `lag` 拍前的数据时，必须保证已经完成
  至少 `lag` 次有效写入和推进。

##### 数据类型使用范围

| 使用形式 | 普通 Python | `@host` | `@host` 内调用的 `@jit` | `@kernel` | `@kernel` 内调用的 `@jit` |
| --- | --- | --- | --- | --- | --- |
| 创建 DelayLineGroup | ✗ | ✗ | ✗ | ✓ | ✓ |
| 调用 DelayLineGroup 或字段视图的方法 | ✗ | ✗ | ✗ | ✓ | ✓ |

✓ 表示支持对应用法；✗ 表示不支持对应用法。

### 属性

构造时声明的每个字段都可以作为 DelayLineGroup 的属性访问。该属性是组内对应字段
的单字段延迟线视图，支持 `push(value)` 和 `tap(lag)`，并与组内其他字段共享
`advance()` 所推进的位置。

```python
dl.<field>.push(value)
dl.<field>.tap(lag)
```

| 属性 | 类型 | 详细说明 |
| --- | --- | --- |
| `dl.<field>` | `DelayLine` 字段视图 | 访问构造时声明的指定字段。该视图只操作本字段，但推进位置由整个 DelayLineGroup 统一管理。 |

#### 属性约束

- 只能访问构造时声明的字段；访问其他字段会失败。
- 字段视图不单独提供 `advance()`；所有字段必须通过组级 `advance()` 一起推进。

## 方法

### `push()`

在当前写入位置同时写入所有字段。字段值按构造时声明的名称匹配，字段书写顺序不影响
匹配结果。

#### 函数原型

```python
dl.push(**kwargs) -> None
```

#### 参数说明

| 参数 | 类型 | 必选 | 默认值 | 简要说明 |
| --- | --- | --- | --- | --- |
| `kwargs` | 整数标量关键字参数 | 是 | 无 | 使用构造时声明的字段名提供当前迭代的字段值。 |

#### 返回值说明

无返回值。

#### 约束说明

- 每次调用必须提供构造时声明的全部字段；缺少任一字段均不支持。
- 每个值必须是 Python `int` 或能够无损转换为 `int64` 的 Kernel 运行时整数。
- `push()` 只写入当前迭代的位置，不会自动调用 `advance()`。

### `tap()`

读取所有字段在 `lag` 次迭代之前写入的值。适用于全部字段在同一流水阶段消费、使用
相同滞后量的场景。

#### 函数原型

```python
dl.tap(lag: int) -> SimpleNamespace
```

#### 参数说明

| 参数 | 类型 | 必选 | 默认值 | 简要说明 |
| --- | --- | --- | --- | --- |
| `lag` | `int` | 是 | 无 | 指定向前读取的迭代次数。 |

##### `lag` 参数可选值

`lag` 必须是 Python `int`，并满足 `0 < lag < depth`。例如，`lag=1` 读取前一次
迭代写入的值，`lag=2` 读取前两次迭代写入的值。

#### 返回值说明

返回 `SimpleNamespace`。其中每个属性对应构造时声明的一个字段，属性值是该字段在
`lag` 次迭代之前写入的值。

#### 约束说明

- 所有字段使用同一个 `lag`。需要为不同字段指定不同 `lag` 时，应使用字段视图的
  `tap()`。
- 调用前必须保证对应历史位置已经写入有效值。

### `<field>.push()`

只向指定字段的当前写入位置写入一个值。该方法适用于字段需要分别产生的场景。

#### 函数原型

```python
dl.<field>.push(value) -> None
```

#### 参数说明

| 参数 | 类型 | 必选 | 默认值 | 简要说明 |
| --- | --- | --- | --- | --- |
| `value` | 整数标量 | 是 | 无 | 写入指定字段当前迭代位置的值。 |

#### 返回值说明

无返回值。

#### 约束说明

- `value` 必须是 Python `int` 或能够无损转换为 `int64` 的 Kernel 运行时整数。
- 调用不会推进位置；完成本次迭代所需字段的写入后，仍需调用组级 `advance()`。

### `<field>.tap()`

读取指定字段在 `lag` 次迭代之前写入的值。不同字段可以使用不同的 `lag`，用于字段
分别在不同流水阶段消费的场景。

#### 函数原型

```python
dl.<field>.tap(lag: int) -> int64
```

#### 参数说明

| 参数 | 类型 | 必选 | 默认值 | 简要说明 |
| --- | --- | --- | --- | --- |
| `lag` | `int` | 是 | 无 | 指定向前读取的迭代次数，必须满足 `0 < lag < depth`。 |

#### 返回值说明

返回指定字段在 `lag` 次迭代之前写入的 `int64` 值。

#### 约束说明

- `lag` 必须是 Python `int`。
- 调用前必须保证对应历史位置已经写入有效值。

### `advance()`

将 DelayLineGroup 的共享推进位置向前移动一次。调用后，全部字段同时切换到下一个
循环位置。

#### 函数原型

```python
dl.advance() -> None
```

#### 参数说明

无参数。

#### 返回值说明

无返回值。

#### 约束说明

- 每次循环迭代结束时调用一次。
- 主循环和流水排空阶段必须采用一致的推进规则；遗漏或重复调用都会使字段值与流水
  阶段错位。

### 调用示例

以下示例同时演示两种读取方式：`dl.tap(1)` 一次读取所有字段，
`dl.<field>.tap(lag)` 分别按不同滞后量读取字段。

```python
import torch
import torch_npu  # noqa: F401

import cannbotdsl as cbd


@cbd.kernel
def dl_bulk_kernel(x: cbd.Tensor, y: cbd.Tensor):
    dl = cbd.DelayLineGroup(3, "row", "val")
    n = x.shape[0]
    for i in range(n):
        dl.push(row=i, val=x[i])
        snap = dl.tap(1)
        if i >= 1:
            y[snap.row] = snap.val * 2
        dl.advance()


@cbd.kernel
def dl_attr_kernel(x: cbd.Tensor, y: cbd.Tensor):
    dl = cbd.DelayLineGroup(4, "a", "b")
    n = x.shape[0]
    for i in range(n):
        dl.push(a=i, b=x[i])
        a1 = dl.a.tap(1)
        b2 = dl.b.tap(2)
        if i >= 2:
            y[a1] = b2
        dl.advance()


class DelayLineGroupDemo:
    @cbd.jit
    def run_bulk(self, x, y):
        dl_bulk_kernel[1](x, y)

    @cbd.jit
    def run_attr(self, x, y):
        dl_attr_kernel[1](x, y)


if __name__ == "__main__":
    n = 8
    x = torch.arange(1, n + 1, dtype=torch.int64, device="npu:0")
    op = DelayLineGroupDemo()

    y = torch.zeros(n, dtype=torch.int64, device="npu:0")
    op.run_bulk(x, y)
    torch.npu.synchronize()
    print("bulk tap     :", y.cpu().tolist())

    y2 = torch.zeros(n, dtype=torch.int64, device="npu:0")
    op.run_attr(x, y2)
    torch.npu.synchronize()
    print("attribute tap:", y2.cpu().tolist())
```

输入：

```text
x = [1, 2, 3, 4, 5, 6, 7, 8]
```

输出：

```text
bulk tap     : [2, 4, 6, 8, 10, 12, 14, 0]
attribute tap: [0, 1, 2, 3, 4, 5, 6, 0]
```

组级读取还可以采用以下字段组合：

```python
dl = DelayLineGroup(3, "batch", "head", "m", "n")
dl.push(batch=b, head=h, m=m, n=n)
snapshot = dl.tap(2)
dl.advance()
```

不同字段使用不同滞后量时，可以采用以下形式：

```python
dl = DelayLineGroup(4, "tile", "n", "phys", "is_last")
dl.push(tile=t, n=n, phys=p, is_last=il)
tile_at_1 = dl.tile.tap(1)
phys_at_2 = dl.phys.tap(2)
last_at_3 = dl.is_last.tap(3)
dl.advance()
```
