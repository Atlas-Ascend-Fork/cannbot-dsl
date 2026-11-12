# 控制流

CANNBot-DSL支持在`@host`、`@jit`和`@kernel`函数中使用条件分支、循环、条件判断和值选择。

## 控制流一览

| 用途 | 支持的写法 | 说明 |
| --- | --- | --- |
| 条件分支 | [if / elif / else](./if.md) | 根据条件选择执行的语句块。 |
| 条件判断 | [and](./predicate-expressions/and.md)、[or](./predicate-expressions/or.md)、[not](./predicate-expressions/not.md)、[链式比较](./predicate-expressions/chained-comparisons.md)、[any](./predicate-expressions/any.md)、[all](./predicate-expressions/all.md) | 产生或组合判断条件。 |
| 值选择 | [条件表达式](./value-selection/conditional-expression.md)、[min](./value-selection/min.md)、[max](./value-selection/max.md) | 根据条件或大小关系选择结果值。 |
| 范围与序列循环 | [for、for ... else](./for.md)、[range](./for/range.md)、[cannbotdsl.range](./for/dsl-range.md)、[range_constexpr](./for/range-constexpr.md) | 按范围或固定序列迭代，支持编译期展开及运行时循环的展开配置。 |
| 条件循环 | [while、while ... else](./while.md) | 根据条件决定是否继续循环。 |
| 编译期控制 | [const_expr](./compile-time-control/const-expr.md) | 在编译期确定条件判断或循环条件的结果。 |

## 约束

以下示例均为核函数体中的代码片段。`cbd`为`cannbotdsl`的导入别名，`flag`为运行时布尔参数，`out`为包含1个`int64`元素的GM输出Tensor。

### 变量只在部分分支中赋值

**错误写法**

```python
if flag:
    value = 1
cbd.scalar.vec_store_bypass(out.ptr(0), value)
```

**报错**

```text
FE201_UNBOUND_REGION_STATE: Region-local value 'value' is not assigned on every path before it is used outside the region.
```

**原因**：当`flag`为假时，`value`没有被赋值，分支后的写入操作无法取得有效值。

**正确写法**：将需要在分支后使用的变量提前初始化：

```python
value = 0
if flag:
    value = 1
cbd.scalar.vec_store_bypass(out.ptr(0), value)
```

### 分支改变变量的容器结构

**错误写法**

```python
values = (cbd.dtypes.int64(0),)
if flag:
    values = [cbd.dtypes.int64(1)]
cbd.scalar.vec_store_bypass(out.ptr(0), values[0])
```

**报错**

```text
FE401_STATE_STRUCTURE_MISMATCH: scf.if then branch structure mismatch: expected tuple[1], got list[1].
```

**原因**：`values`在分支前是tuple，分支中被改成list。跨运行时分支或循环更新变量时，应保持容器种类、长度以及字典键的顺序一致。

**正确写法**：本例统一使用tuple：

```python
values = (cbd.dtypes.int64(0),)
if flag:
    values = (cbd.dtypes.int64(1),)
cbd.scalar.vec_store_bypass(out.ptr(0), values[0])
```

### 分支改变变量的数据类型

**错误写法**

```python
value = cbd.dtypes.int64(0)
if flag:
    value = cbd.dtypes.float32(1.0)
cbd.scalar.vec_store_bypass(out.ptr(0), value)
```

**报错**

```text
FE402_STATE_TYPE_MISMATCH: scf.if then branch changed MLIR leaf 0 from i64 to f32.
```

**原因**：`value`从`int64`变成`float32`，不满足跨分支传递时的类型要求。

**正确写法**：本例应保持为`int64`：

```python
value = cbd.dtypes.int64(0)
if flag:
    value = cbd.dtypes.int64(1)
cbd.scalar.vec_store_bypass(out.ptr(0), value)
```

需要转换计算结果时，应显式转换为该变量原有的数据类型。类型说明见[数据类型](../../data-structures/data-types.md)。

### 在分支外使用分支内定义的局部Tensor

**错误写法**

```python
if flag:
    local = out[0:1]
else:
    local = out[0:1]
cbd.scalar.vec_store_bypass(local.ptr(0), 1)
```

**报错**

```text
FE202_REGION_VALUE_ESCAPE: Region-local value 'local' cannot escape a dynamic if region.
```

**原因**：两条分支都定义了`local`，但它仍是运行时分支内定义的局部变量，不能直接在分支外使用。

**正确写法**：需要在后续代码中使用该Tensor视图时，将其定义移到分支外，再在各分支中使用：

```python
local = out[0:1]
if flag:
    cbd.scalar.vec_store_bypass(local.ptr(0), 1)
else:
    cbd.scalar.vec_store_bypass(local.ptr(0), 1)
```

### 在循环外使用循环变量

**错误写法**

```python
for i in cbd.range(1):
    pass
cbd.scalar.vec_store_bypass(out.ptr(0), i)
```

**报错**

```text
FE202_REGION_VALUE_ESCAPE: Loop induction variable 'i' is region-local and cannot be used after the loop.
```

**原因**：运行时`for`的循环变量只能在循环体内使用。

**正确写法**：需要保留循环中的值时，使用循环前已初始化的变量：

```python
value = 0
for i in cbd.range(1):
    value = i
cbd.scalar.vec_store_bypass(out.ptr(0), value)
```

### 使用break退出运行时循环

**错误写法**

```python
for i in cbd.range(1):
    break
cbd.scalar.vec_store_bypass(out.ptr(0), 0)
```

**报错**

```text
FE102_UNSAFE_EARLY_EXIT: break cannot cross the boundary of a dynamic control-flow region.
```

**原因**：运行时循环不支持通过`break`或`continue`跳出或跳过迭代，运行时分支或循环中也不支持使用`return`、`raise`提前退出。

**正确写法**：本例可直接限定循环范围，在循环体中完成写入：

```python
for i in cbd.range(1):
    cbd.scalar.vec_store_bypass(out.ptr(0), i)
```

编译期展开的循环可使用`break`、`continue`控制自身循环；在运行时控制流中嵌套编译期循环时，这些语句也只能作用于内层编译期循环。不同循环形式见[for](./for.md)。

### 编译期条件依赖运行时数据

**错误写法**

```python
if cbd.const_expr(flag):
    cbd.scalar.vec_store_bypass(out.ptr(0), 1)
```

**报错**

```text
TypeError: const_expr() requires a compile-time Python value; use an if statement inside a @host/@jit/@kernel boundary for a runtime DSL predicate
```

**原因**：`flag`是运行时参数，编译时无法确定其值。

**正确写法**：需要根据运行时数据选择分支时，直接使用`if`：

```python
if flag:
    cbd.scalar.vec_store_bypass(out.ptr(0), 1)
```

只有编译期已知的条件才使用[const_expr](./compile-time-control/const-expr.md)。编译期循环范围的要求见[range_constexpr](./for/range-constexpr.md)。
