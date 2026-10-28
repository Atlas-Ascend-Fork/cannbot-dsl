# 矩阵计算

矩阵计算接口完成Mmad矩阵乘加及其计算模式配置，仅在AIC上生效。

- **[概述](./overview.md)**
- **[矩阵计算单元](./unit.md)**
- **[计算流程](./flow.md)**
- **[背景与核心概念](./fractal-intro.md)**
- **[关键分形格式](./fractal-formats.md)**
- **[矩阵计算关键特性说明](./key-features.md)**
  - **[GEMV](./key-features-gemv.md)**
  - **[HF32](./key-features-hf32.md)**
  - **[UnitFlag](./key-features-unit-flag.md)**

## 接口

- **[`enable_fp8`](./enable-fp8.md)**
- **[`enable_hf32`](./enable-hf32.md)**
- **[`enable_hif8`](./enable-hif8.md)**
- **[`matmul`](./matmul.md)**
- **[`set_fp32_mode`](./set-fp32-mode.md)**
- **[`set_hf32_round_mode`](./set-hf32-round-mode.md)**
- **[`set_mmad_direction`](./set-mmad-direction.md)**
