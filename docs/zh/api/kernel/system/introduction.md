# Kernel API：系统变量访问


系统变量访问 API 用于在设备 Kernel 中查询任务索引、物理核、子核、系统周期和状态寄存器等运行时信息。

## 接口

| 接口 | 说明 |
| --- | --- |
| [`get_block_idx`](/zh/api/kernel/system/get-block-idx) | 获取当前 block 索引。 |
| [`get_block_num`](/zh/api/kernel/system/get-block-num) | 获取本次 Kernel 启动的 block 数量。 |
| [`get_subblock_id`](/zh/api/kernel/system/get-subblock-id) | 获取当前子核索引。 |
| [`get_subblock_dim`](/zh/api/kernel/system/get-subblock-dim) | 获取当前类型的子核数量。 |
| [`get_core_id`](/zh/api/kernel/system/get-core-id) | 获取当前物理核 ID。 |
| [`get_system_cycle`](/zh/api/kernel/system/get-system-cycle) | 读取系统时钟周期计数。 |
| [`get_status`](/zh/api/kernel/system/get-status) | 读取状态寄存器。 |
| [`get_vf_len`](/zh/api/kernel/system/get-vf-len) | 获取向量函数一次处理的向量长度。 |
| [`get_squeeze_status`](/zh/api/kernel/system/get-squeeze-status) | 获取 squeeze 有效数据长度。 |
