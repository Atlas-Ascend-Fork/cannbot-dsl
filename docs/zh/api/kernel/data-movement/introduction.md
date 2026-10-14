# Kernel API：数据搬运


数据搬运 API 用于在 GM、UB、L1、L0A、L0B、L0C 等存储层级之间传输数据。

## 接口

- [`make_copy_engine`](/zh/api/kernel/data-movement/make-copy-engine)：创建供 `mem_copy` 使用的数据搬运配置。
- [`mem_copy`](/zh/api/kernel/data-movement/mem-copy)：在 Tensor 之间搬运数据。
