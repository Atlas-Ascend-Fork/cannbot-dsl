# Kernel API：同步与缓存控制

AI Core 的标量计算单元直接读写全局内存（Global Memory，GM）时，数据可能暂存在每个核内部的数据缓存（Data Cache，DCache）中。DCache 以缓存行（Cache Line）为最小管理单位；将缓存行“写回”表示把修改同步到 GM，将其“失效”表示下次访问时重新从 GM 读取。

这些接口用于保证 DCache 与 GM 中的数据一致。通过直接内存访问（Direct Memory Access，DMA）在 GM 与片上存储之间搬运数据时不经过 DCache，通常不需要调用这些接口。

> 这些接口不会由框架根据 GM 访问自动插入。使用 Scalar 直接读写 GM 时，调用方必须根据数据共享关系显式选择 `dcci_*`、`dci` 或 bypass 接口。

- **[概述](/zh/api/kernel/synchronization-cache/)**

## 接口

| 接口 | 说明 |
| --- | --- |
| [`dcci_single`](/zh/api/kernel/synchronization-cache/dcci-single) | 写回并失效指定 GM 地址所在的一个缓存行。 |
| [`dcci_entire_out`](/zh/api/kernel/synchronization-cache/dcci-entire-out) | 写回并失效本核数据缓存中的所有 GM 数据。 |
| [`dcci_entire_atomic`](/zh/api/kernel/synchronization-cache/dcci-entire-atomic) | 写回并失效标量原子操作访问的 GM 数据。 |
| [`dci`](/zh/api/kernel/synchronization-cache/dci) | 失效本核的整个数据缓存，并丢弃尚未写回的修改。 |
