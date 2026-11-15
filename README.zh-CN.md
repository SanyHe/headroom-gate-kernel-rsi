# Headroom-Gate Kernel RSI

一个可被证伪的 GPU kernel 评估闸门：优化之前先把上限算出来，测量之后让一次留出判决来收尾。

大多数 kernel 文章都朝同一个方向失真——上限从未被写出来，于是任何提升都像进展。这个仓库把上限变成输入。五个预注册案例（1 个 GO、4 个 NO-GO）在没有 GPU 的笔记本上复算出归档判决，因为闸门只是对已记录测量做的纯算术。

[English](./README.md) · **简体中文** · [结果看板](https://sanyhe.github.io/headroom-gate-kernel-rsi/)

## 一屏看懂闸门

1. **字节。** 汇总工作负载的必需流量（权重 + 激活 + 输出），这一项无法被论证掉。
2. **地板。** `max(bytes / BW_eff, flops / FLOP_eff)`；H20 主机上 3.30 TB/s 与 138 TFLOPS 都是实测值，不是规格表。可选的同字节参考 kernel 只能抬高地板，不能压低。
3. **空间。** `实测中位数 - 地板` 是唯一存在的余量。
4. **可寻址。** `min(声称比例 × 余量, 可移除字节 / BW, 可移除毫秒)`。机制移除的时间不能超过它触碰的字节或时间。
5. **区间。** 对重复样本做 5–95% bootstrap。区间内出现拐点会显式求值：一旦上限生效，`speedup(t)` 不再单调。
6. **判决。** 只有保守端 ≥ 1.30× **且**存在同字节实测地板才 `GO`；宽松端 < 1.15× 则 `NO-GO`；其余为 `INCONCLUSIVE`——这是诚实的答案，不是失败。

```bash
python3 -m gate.cli --all
```

| 案例 | 判决 | 上限（5–95%，点估计） | 原因 |
|---|---|---|---|
| `swiglu-fusion-b4096-h7168` | **GO** | 4.993–5.004×（4.999×） | 176 MB 流式读写融合为一遍；同样字节的 `torch.add` 只要 0.0531 ms |
| `attn-decode-longctx` | NO-GO | 1.022–1.035× | 读取 128 MiB KV；同一缓存的求和已需 0.0614 ms，而实测 0.0637 ms |
| `gemv-m8-bf16` | NO-GO | 1.022–1.048×（1.030×） | 200 MiB 权重流；最好情形就是流式读一遍，约 0.0803 ms |
| `selector-dflash2` | NO-GO | 1.0048–1.0051× | 22.68 ms 一步里的 0.11 ms，且被时间账本封顶 |
| `verify-fusion-megaspec` | NO-GO | 1.000–1.003× | 可避免流量只占 17.23 ms 中的 0.056 ms |

四个 NO-GO 才是这份工作真正的产出：每一个都是业界常做的优化，但在写下第一行 kernel 之前就被算出「付不起」。

## 快速开始

```bash
./init.sh                                        # 建 venv、复算判决、跑测试，无需 GPU
python3 -m gate.cli cases/attn-decode-longctx     # 单案例 -> gate_verdict.json
python3 -m gate.cli --all --record                # 判决写入 ledger.jsonl
python3 -m gate.cli --verify-ledger               # 主张 P1：realised <= predicted
```

依赖：Python 3.10+、`PyYAML`、`pytest`，没有别的。

## 判决承诺什么

- **P1 棘轮。** 任何实测加速不得超过预测上限；由 `ledger.jsonl` 机械检查。
- **P2 不相交。** 评估算子不出现在测量案例里；按构造保证。封存运行器已落地并在第二台主机上运行过，事后冻结已注明。
- **范围可证。** `out_of_scope_checks`（精度、模型、接受率、输出分布）触发即判 `INVALID`，不进行口头讨论。
- **身份可复现。** 每个判决内嵌 `protocol/gate_rules.yaml` 的 sha256；漂移会被发现，而不是被讨论。

## 当前状态：哪些能跑

本地已实现并验证（无需 GPU）：

| 模块 | 位置 |
|---|---|
| 冻结规则 + 哈希校验 | `protocol/gate_rules.yaml`、`gate/rules.py` |
| 字节/FLOP 账本、roofline、经验地板折叠 | `gate/byte_model.py` |
| 判决逻辑、拐点感知的上下界 | `gate/decide.py` |
| bootstrap / 参数化区间、配对检验 | `stats/bootstrap.py` |
| 追加式账本 + P1 棘轮 | `gate/ledger.py` |
| CLI 与判决产物 | `gate/cli.py` |
| 五个预注册案例 | `cases/*/case.yaml` |
| 后台运行契约（detach / status / stop / resume） | `scripts/` |
| 冻结的 P3 策略与确定性选择器 | `p3/policy.md`、`p3/select.py` |
| 71 个 CPU 测试 | `tests/` |

在第二台主机（RTX 6000D，v2 常数）上实测：算子层 Triton 1.92×、CUDA 2.12×，输出逐位一致；
留出形状保持 1.88× / 1.30× / 1.72×；P1 棘轮未违规。模型层同一融合约为 1.00×：
4096 token 的 prefill 是发射受限的，省下的 8.3 ms 激活 GPU 时间不会改变 506 ms 的一步。
这就是对外发布的边界条件——见 [03 — 评估](./docs/03-eval.md#6-第二台主机一个边界条件结果)。
仍待完成：注册过的 P3 选择（已有探索性计时；池中没有精度 trial）。

因此，五个归档判决是从 `case.yaml` 里记录的 `measured_ms` 复算出来的，没有重新测量。
所用样本全部内嵌在案例文件中；原始归档为私有，不再分发。`case.yaml` 只追加、不改写；
见 [`docs/02-measurement.md`](./docs/02-measurement.md)。

## 目录

```
protocol/gate_rules.yaml   冻结阈值、设备常数、范围
gate/                      字节模型、判决、账本、CLI、设备探针
stats/bootstrap.py         中位数 CI、参数化区间、配对 bootstrap
cases/<id>/case.yaml       预注册假设、字节数、测量
cases/<id>/gate_verdict.json  生成物，不手改
p3/                        冻结选择策略 + 确定性选择器
scripts/                   后台运行契约（以 status.json 为准）
docs/                      方法、测量、评估、P3
private/                   harness 状态（gitignore）
```

## 文档

- [01 — 闸门](./docs/01-gate.md)：账本、地板、裁剪、区间、判决。
- [02 — 测量](./docs/02-measurement.md)：计时协议与后台运行契约。
- [03 — 评估](./docs/03-eval.md)：P1/P2、五个复算、第二台主机的边界条件，以及本仓库唯一一处诚实的偏差。
- [04 — P3](./docs/04-p3.md)：预注册选择与联合影响规则。

## 许可

Apache-2.0，见 [LICENSE](./LICENSE)。
