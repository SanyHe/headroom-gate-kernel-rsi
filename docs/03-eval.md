# 03 — 评估：两个主张、一个实测案例、一处诚实偏差

## 1. 两个主张

**P1 棘轮（可机械检查）：** 任何实测加速都不得超过当时判决给出的预测上限。
`gate/cli.py --verify-ledger` 从账本重建每个案例的"最近一次 gate 判决"，
把每条 `realised` 事件的 `realised_speedup` 与之比较，超出即报告违规并返回码 4。

严格说，这条主张要求 P1 先于 P2 成立：先预测，再测量，顺序不可交换。

**P2 不相交（构造性检查）：** 评估算子不得出现在测量案例中。评估集在
`protocol/gate_rules.yaml` 之外的封存清单里维护，`sealed_eval.py` 只跑封存清单，
`cases/` 里不允许出现同名算子。GPU 评估脚本已经落地，并在 AutoDL RTX 6000D 上运行过；
但 manifest 在候选已存在之后才冻结，因此那次运行是留出形状复核，不是盲测封存，
见第 5 节。

## 2. 实测案例的本地复算

```bash
python3 -m gate.cli cases-rtx/swiglu-fusion-b4096-h7168-rtx6000d
python3 -m pytest tests/test_gate.py -q
```

| 案例 | 判决 | 上限 5–95% | 点估计 | 地板来源 |
|---|---|---|---|---|
| `swiglu-fusion-b4096-h7168-rtx6000d` | **GO** | 2.147 – 2.155 | 2.151 | 经验 0.10834（解析 0.10910，经验更高） |

判决从 `case.yaml` 里记录的 `measured_ms` 逐位复算出来。`cases/` 下的五个模板案例不含样本，不产生判决；它们是预注册的设计，等待三方云测量。

## 3. 测试覆盖

`python3 -m pytest tests -q` 共 71 个用例：

| 文件 | 覆盖 |
|---|---|
| `tests/test_gate.py` | 实测案例复算、拐点求值、越界/缺字段/哈希不符/机制类非法、单样本路径 |
| `tests/test_byte_model.py` | 字节账本、roofline 区制、经验地板只抬不降、负例 |
| `tests/test_bootstrap.py` | 中位数/百分位、bootstrap 确定性、参数化区间保守端、配对检验 |
| `tests/test_ledger.py` | 追加写、损坏行定位、状态重建、P1 违规检测 |
| `tests/test_cli.py` | 单案例/全量、`--record`、`--verify-ledger` 返回码、INVALID 退出码 3 |
| `tests/test_runmeta.py` | init/status/heartbeat/checkpoint、原子写、字段保留 |
| `tests/test_p3_select.py` | R1–R4、联合测量要求、策略哈希冻结与篡改检测 |
| `tests/test_probes.py` | 设备探针计划在无 GPU 情况下可打印，且不导入 torch |

## 4. 复现边界

本地能复算判决，不能复算"测量"。实测案例的判决依赖其 `case.yaml` 里的样本；这些样本来自 AutoDL RTX 6000D 主机，所用样本已全部内嵌在案例文件中，原始证据位于 `cases-rtx/<id>/evidence/`。`cases/` 下的五个模板案例不含样本，在 `blocked` 的 GPU 脚本于三方云硬件上运行之前无法复现测量；见 [02 — 测量](./02-measurement.md)。

## 5. AutoDL RTX 6000D 实测案例：一个边界条件结果

GO 案例按 `protocol/gate_rules.yaml`（版本 2.0.0，`hbm_realistic_tbs = 1.613`）在 AutoDL RTX 6000D 上注册，然后做了端到端测量。下面所有数字都来自这台设备；每次运行各自记录为独立产物，位于 `cases-rtx/swiglu-fusion-b4096-h7168-rtx6000d/`。

| 阶段 | 结果 |
|---|---|
| 闸门（P1） | 预测 2.1515×（2.1474–2.1545），实测 1.91595×（1.8997–1.9308），比值 0.89 |
| 留出形状 | 1.876×（b8192-h7168）、1.299×（b1024-h7168）、1.720×（b4096-h5120），全部逐位一致 |
| 常数交叉验证 | 4096×7168、4096×12288、8192×7168 上为 1.928× / 1.917× / 1.895×；控制组 0.74–0.76 TB/s |
| 三臂 dispatch | 四个形状上 shipped → CUDA 为 2.120× / 2.005× / 1.654× / 1.936×，三臂全部逐位一致 |
| 模型 prefill，4096 token | Triton 1.0031×，CUDA 0.9999×；128 个 logits 逐位一致 |
| 模型 decode，b=1，32 步 | Triton 0.9563×，CUDA 1.0177× |
| 模型 decode，b=256，32 步 | Triton 1.0021×，CUDA 1.0012× |

算子层的收益真实且可复现，但没有变成模型层的收益；测量本身给出了原因：prefill 这一步是发射受限，而不是带宽受限。

| 每次 4096-token prefill | 现有路径 | 融合（CUDA） |
|---|---|---|
| 激活 GPU 时间（36 个 MLP） | 16.1 ms | 7.8 ms |
| CPU 入队 | 239 ms | 237 ms |
| GPU 排空 | 267 ms | 269 ms |
| Wall | 506 ms | 506 ms |

把激活整个删掉的消融仍需 505–506 ms。只做 CPU 的 profile 显示，这一步有 227 ms 落在未融合 RMSNorm 的 `aten::rsqrt`、`aten::pow`、`aten::mean` 上。省下 8.3 ms GPU 时间，无法缩短一步关键路径在 CPU 入队的计算。隔离基准里同一个算子冷态仍有 1.99×、热态 1.86×，因此模型层的负结果不是测量假象。

**因此对外发布的主张是一个边界条件：** 融合 `silu_and_mul` 只有在周围步骤是 GPU 受限时才会兑现，它无法改造一个发射受限的模型。算子层天花板成立；模型层加速如实记为约 1.00×。

这个结果旁边必须同时写明两处偏差：

- 封存 manifest 在候选存在之后才冻结（`frozen_after_candidate: true`）：这是留出形状复核，不是盲测封存。
- 第三臂是 vLLM 预编译的 `torch.ops._C.silu_and_mul`，因为主机没有 `nvcc`。它是真实的编译算子，但不是本仓库自己编译的。

P3 没有注册过的选择：`p3/evidence/` 里有探索性计时，但池中没有低比特臂，也没有精度 trial，因此不声明任何低比特结论。
