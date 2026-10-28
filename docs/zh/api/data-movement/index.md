# 数据搬运


数据搬运接口用于在 GM、UB、L1、L0A、L0B、L0C 等存储层级之间传输数据。

## 接口

- [`copy_options`](./copy-options.md)：创建或派生供 `mem_copy` 使用的数据搬运配置。
- [CopyOptions](./copy-options-type.md)：数据搬运配置类型。
- [`mem_copy`](./mem-copy.md)：在 Tensor 之间搬运数据。
