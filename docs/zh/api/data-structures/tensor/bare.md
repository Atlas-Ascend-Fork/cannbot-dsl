# Tensor.bare

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

获取当前 Tensor 绑定的 Storage 句柄，用于查看容量或绑定其他布局。

## 函数原型

```python
bare(self) -> Storage
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
无。

## 返回值说明

返回当前绑定对应的 [Storage](../storage/storage/overview.md)，可读取 `size_bytes`、`alignment` 和 `memloc`。

## 约束说明

- 仅限携带存储绑定的 Tensor，例如 Buffer/make_buffer 返回值、RingBuffer 选槽结果以及 Storage.view 返回值。
- 任意切片或变换后的 Tensor 不保证保留该句柄；需要复用句柄时，在派生视图前保存 `t.bare()` 的结果。

## 调用示例

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def example_kernel(y: cbd.Tensor):
    t = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((32,)))
    s = t.bare()
    u = s.view(cbd.make_layout((2, 8)), offset=32)
    y[0] = s.size_bytes
    y[1] = s.alignment
    y[2] = u.shape[0]
    y[3] = u.shape[1]


@cbd.host
def run(y):
    example_kernel[1](y)


y = torch.empty(4, dtype=torch.int64, device="npu")
run(y)
torch.npu.synchronize()
print("存储字节数:", y[0].item())
print("对齐字节数:", y[1].item())
print("视图 shape:", y[2:].cpu().tolist())
```

输入：

```text
Buffer: mem_loc=UB, dtype=float32, layout=make_layout((32,))
新视图 layout=make_layout((2, 8)), offset=32
```

输出：

```text
存储字节数: 128
对齐字节数: 32
视图 shape: [2, 8]
```
