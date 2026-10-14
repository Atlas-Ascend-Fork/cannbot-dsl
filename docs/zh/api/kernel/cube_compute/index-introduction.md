# 矩阵计算

矩阵计算接口完成 Mmad 矩阵乘加及其计算模式配置，仅在 AIC 上生效。

- **[概述](/zh/api/kernel/cube_compute/)**
- **[矩阵计算单元](/zh/api/kernel/cube_compute/unit)**
- **[计算流程](/zh/api/kernel/cube_compute/flow)**
- **[背景与核心概念](/zh/api/kernel/cube_compute/fractal-intro)**
- **[关键分形格式](/zh/api/kernel/cube_compute/fractal-formats)**
- **[矩阵计算关键特性说明](/zh/api/kernel/cube_compute/key-features)**
  - **[GEMV](/zh/api/kernel/cube_compute/key-features-gemv)**
  - **[HF32](/zh/api/kernel/cube_compute/key-features-hf32)**
  - **[UnitFlag](/zh/api/kernel/cube_compute/key-features-unit-flag)**

## 接口

- **[`enable_fp8`](/zh/api/kernel/cube_compute/enable-fp8)**
- **[`enable_hf32`](/zh/api/kernel/cube_compute/enable-hf32)**
- **[`enable_hif8`](/zh/api/kernel/cube_compute/enable-hif8)**
- **[`matmul`](/zh/api/kernel/cube_compute/matmul)**
- **[`set_fp32_mode`](/zh/api/kernel/cube_compute/set-fp32-mode)**
- **[`set_hf32_round_mode`](/zh/api/kernel/cube_compute/set-hf32-round-mode)**
- **[`set_mmad_direction`](/zh/api/kernel/cube_compute/set-mmad-direction)**
