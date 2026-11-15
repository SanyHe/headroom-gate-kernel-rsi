# 03 — 评估：两个主张、五个复算、一处诚实偏差

## 1. 两个主张

**P1 棘轮（可机械检查）：** 任何实测加速都不得超过当时判决给出的预测上限。
`gate/cli.py --verify-ledger` 从账本重建每个案例的“最近一次 gate 判决”，
把每条 `realised` 事件的 `realised_speedup` 与之比较，超出即报告违规并返回码 4。

严格说，这条主张要求 P1 先于 P2 成立：先预测，再测量，顺序不可交换。

**P2 不相交（构造性检查）：** 评估算子不得出现在测量案例中。评估集在
`protocol/gate_rules.yaml` 之外的封存清单里维护，`sealed_eval.py` 只跑封存清单，
`cases/` 里不允许出现同名算子。GPU 评估脚本尚未落地，因此 P2 目前是“已声明、待执行”，
不是“已通过”。

## 2. 五个归档案例的本地复算

```bash
python3 -m gate.cli --all
python3 -m pytest tests/test_gate.py -q
```

| 案例 | 判决 | 上限 5–95% | 点估计 | 地板来源 |
|---|---|---|---|---|
| `swiglu-fusion-b4096-h7168` | **GO** | 4.993443 – 5.004346 | 4.999138 | 解析 0.053382（经验 0.053104 更低，被忽略） |
| `attn-decode-longctx` | NO-GO | 1.021620 – 1.035426 | 1.035426 | 经验 0.061424（解析 0.040675 偏低） |
| `gemv-m8-bf16` | NO-GO | 1.021552 – 1.048246 | 1.029720 | 经验 0.080317（解析 0.063550） |
| `selector-dflash2` | NO-GO | 1.004749 – 1.005068 | 1.004906 | 解析（单样本，参数化区间） |
| `verify-fusion-megaspec` | NO-GO | 1.000000 – 1.003320 | 1.003257 | 解析（单样本，参数化区间） |

这些数字与归档判决逐位一致。它们是整个仓库的回归基线：
`tests/test_gate.py::test_archived_verdicts_replay` 以 1e-4 容差断言全部 15 个数值。

四个 NO-GO 的解读方式值得固定下来：

- `attn-decode-longctx`：解析 roofline 给出 1.56× 的“余量”，但同字节求和只需 0.0614 ms，
  实测 0.0637 ms——真实余量 3.4%。假设被自己的测量证伪，`post_measurement_note` 保留原假设。
- `gemv-m8-bf16`：0.1% 的余量，低于 2% 最小余量门槛，且没有任何字节可删。
- 两个端到端案例：机制只占整步的 0.3–0.5%，而且延迟主导，字节账本封顶比时间账本更严。

## 3. 唯一一处与归档不一致：规则文件哈希

归档记录 `protocol/gate_rules.yaml` 的 sha256 为 `6280dcff…`。本仓库重新落盘后的实际
哈希是 `966715a7dfe9df58e71175719fa3f62226d474e368adc42c177fe7d17416bea2`。

原因：文档拷贝过程中空白/换行无法按字节还原，原文件的精确字节不可恢复。处理方式遵循
规格 §4.1 —— **只改身份哈希，不改任何数值**：

- 五个 `case.yaml` 的 `rules_sha256` 重钉到 `966715a7…`；
- 阈值、设备常数、机制声称、`measured_ms` 一律未动；
- 判决与 15 个数值逐位一致（见上表）；
- `tests/test_gate.py` 同时断言“当前哈希 == 重钉值”和“每个案例固定同一个哈希”，
  所以后续任何规则改动都会立刻让测试变红。

这是本仓库唯一需要读者知道的不一致点，写在这里而不是藏在提交信息里。

## 4. 测试覆盖

`python3 -m pytest tests -q` 共 71 个用例：

| 文件 | 覆盖 |
|---|---|
| `tests/test_gate.py` | 五个归档判决复算、拐点求值、越界/缺字段/哈希不符/机制类非法、单样本路径 |
| `tests/test_byte_model.py` | 字节账本、roofline 区制、经验地板只抬不降、负例 |
| `tests/test_bootstrap.py` | 中位数/百分位、bootstrap 确定性、参数化区间保守端、配对检验 |
| `tests/test_ledger.py` | 追加写、损坏行定位、状态重建、P1 违规检测 |
| `tests/test_cli.py` | 单案例/全量、`--record`、`--verify-ledger` 返回码、INVALID 退出码 3 |
| `tests/test_runmeta.py` | init/status/heartbeat/checkpoint、原子写、字段保留 |
| `tests/test_p3_select.py` | R1–R4、联合测量要求、策略哈希冻结与篡改检测 |
| `tests/test_probes.py` | 设备探针计划在无 GPU 情况下可打印，且不导入 torch |

## 5. 复现边界

本地能复算判决，不能复算“测量”。判决依赖 `case.yaml` 里的样本；这些样本来自 H20 主机上
的原始采集，所用样本已全部内嵌在案例文件中，原始归档为私有、不再分发。要重新产生样本，
需要把 `blocked` 的 GPU 脚本落地，并按
[02 — 测量](./02-measurement.md) 的协议重跑。
