# GroupSync.release

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

提交一次消费者完成通知，允许后续轮次复用槽位。消费者应在完成该轮数据读取后调用。

## 函数原型

```python
release(self, ticket)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `ticket` | 输入 | 通过本对象的`wait`成功取得的消费者凭据。 |

## 返回值说明

无返回值。

## 约束说明

- `q`为非负的轮次编号。使用同一槽位的各参与者须按一致的轮次推进，并保证计数不溢出。
- 凭据只能由对应的`GroupSync`对象按生产者或消费者角色使用一次，不能作为循环状态或独立函数的ABI参数传递。
- 使用非阻塞形式时，仅在`ready`为真时使用凭据；失败后可再次调用检查。
- 接口不会自动完成有效数据的内存屏障或缓存维护；调用方须根据数据访问方式保证可见性和执行顺序。

其他使用要求见[GroupSync](../group-sync.md)。

## 调用示例

以下示例使用非阻塞形式完成一轮生产和消费，并在Host侧检查两次操作均已就绪。

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def kernel(workspace: cbd.Tensor, result: cbd.Tensor):
    sync = cbd.GroupSync(workspace, depth=1, producers=1, consumers=1)
    ready, ticket = sync.acquire(0, blocking=False)
    cbd.scalar.vec_store_bypass(result.ptr(0), cbd.dtypes.int32(ready))
    if ready:
        cbd.data_barrier(scope="gm")
        sync.commit(ticket)
        cbd.data_barrier(scope="gm")
    ready, ticket = sync.wait(0, blocking=False)
    cbd.scalar.vec_store_bypass(result.ptr(1), cbd.dtypes.int32(ready))
    if ready:
        cbd.data_barrier(scope="gm")
        sync.release(ticket)
        cbd.data_barrier(scope="gm")


@cbd.host
def run(workspace, result):
    kernel[1](workspace, result)


workspace = torch.zeros((16,), dtype=torch.int64, device="npu")
result = torch.full((2,), -1, dtype=torch.int32, device="npu")
run(workspace, result)
assert result.cpu().tolist() == [1, 1]
```
