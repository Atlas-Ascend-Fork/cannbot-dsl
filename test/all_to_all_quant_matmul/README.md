# AllToAllQuantMatmul 测试目录

## 1. 目录内容

| 文件 | 用途 |
|---|---|
| `test_all_to_all_quant_matmul.py` | 正确性测试（七钩子 + pytest L0 用例，golden 内嵌；多卡批跑 runner） |
| `test_all_to_all_quant_matmul_perf.py` | 性能驱动（精度门禁 + 多轮计时，`--cases` 按名选择） |
| `cases.csv` | 用例表：首列留空 = 正确性用例（多 dtype / bias / rank_size 组合），首列带名 = 性能用例 |

## 2. 正确性测试

```bash
# L0 无卡单测（AST 校验 + codegen 断言）
pytest test/all_to_all_quant_matmul/test_all_to_all_quant_matmul.py -v

# L1 多卡精度（正确性用例 = cases.csv 首列留空的行；2 卡改 --devices 0,1）
python3 test/all_to_all_quant_matmul/test_all_to_all_quant_matmul.py \
    --csv test/all_to_all_quant_matmul/cases.csv --devices 0,1,2,3 --comm-mode urma

# 两阶段：无卡编译发布 bundle → 设备空闲时执行
python3 test/all_to_all_quant_matmul/test_all_to_all_quant_matmul.py \
    --csv test/all_to_all_quant_matmul/cases.csv --stage compile \
    --bundle-root ./bundles --comm-mode urma
python3 test/all_to_all_quant_matmul/test_all_to_all_quant_matmul.py \
    --csv test/all_to_all_quant_matmul/cases.csv --stage execute \
    --bundle-root ./bundles --devices 0,1 --comm-mode urma

# 连续 launch 一致性（--launches > 1）
python3 test/all_to_all_quant_matmul/test_all_to_all_quant_matmul.py \
    --csv test/all_to_all_quant_matmul/cases.csv --devices 0,1 --comm-mode ub \
    --launches 2
```

runner 能力（批跑提效）：

- **设备池语义**：`--devices` 声明卡池，rank_size ≤ 池大小的用例自动取前
  rank_size 张卡——混排 2/4 卡的 cases.csv 一次 launch 全量跑完。
- **golden/b 去冗余**：golden rank 不变（每卡全量输出）时整 case 只算一份；
  b/bias 同 case 内各 rank 共享一份。
- **编译签名 memo**：同进程内相同编译签名（tb/launches 不参与编译）复用
  首次新鲜编译产物，同时规避 DSL 内存缓存命中路径（该路径与 HCCL init
  组合会死锁）。
- **两阶段 bundle**：`--stage compile` 以 `ProviderCallable.save` 发布
  .so bundle；`--stage execute` 的 worker 在设备初始化前自行
  `cannbotdsl.load` 恢复，父进程不再重编译；`cannbotdsl.load` 不可用时
  自动降级为父进程编译。
- **case 看门狗**：`--case-timeout`（默认 600s，0 关闭）内未完成的 case
  判失败并 TERM/KILL 收割卡死 worker，批跑继续；阶段一失败不广播
  START_DEVICE，避免存活 rank 挂死在集合通信里。

### `--persistent-domain` 常驻通信域模式

单通信域常驻批跑：按 rank_size 分域，worker 引导一次通信域后循环执行
CASE_GO 下发的用例（bundle 逐 case 加载），case 失败 TERM/KILL 收割并
重建域。前置条件为 kernel 的跨 launch 同步协议（launch 计数器 ack 屏障
+ 固定就绪 flag），协议不满足时跨 case 会损坏数据。

## 3. 性能测试

```bash
# --cases 按性能用例名选择（cases.csv 首列带名的行），缺省跑全部
python3 test/all_to_all_quant_matmul/test_all_to_all_quant_matmul_perf.py \
    --csv test/all_to_all_quant_matmul/cases.csv --devices 0,1,2,3 \
    --comm-mode urma --cases case1
```

- **精度门禁**：驱动自动比对 golden（rtol=atol=0.01），任一 rank FAIL
  则该 case 性能数据作废。
- **计时**：每轮先入队重型 8192³ fp16 matmul 吸收 host 下发抖动，再
  `[ev0 → 被测 kernel → ev1]` 全部预入队、末尾一次 synchronize；NPU
  event 取纯设备时长，**10 轮取后 5 轮平均**（`--rounds` 可调，≥6）。
- **口径**：每 rank 独立输出 `last5_avg_us`；case 性能 = 各 rank 的
  last5 均值取 max（µs）。
- 注：`case_32m` / `case_64m`（K=16384/32768）在 fp16 输出 + 当前测试
  数据域下元素和可超 65504，精度门禁会 FAIL，属数据域伪影、非算子缺陷。
