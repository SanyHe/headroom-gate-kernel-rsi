# 01 — 闸门：先用算术定价，再动 kernel

闸门是 `gate/decide.py` 里的一个纯函数：输入 `(case, rules)`，输出一个判决字典。
它不读时钟、不碰 GPU、不写文件，所以归档案例任何时候都能复算出同一个判决。

## 1. 输入：一个预注册案例

`cases/<id>/case.yaml` 在测量之前写死这些字段：

| 字段 | 含义 |
|---|---|
| `workload.bytes` | 必需流量，按 `weights / activations / output` 分项 |
| `workload.flops` | 已发生（不是理论）的浮点运算量 |
| `workload.measured_ms` | 逐次实测样本；中位数与区间都从这里来 |
| `workload.empirical_floor_ms` | 同样字节的参考 kernel 的实测耗时（可选，但 GO 必需） |
| `mechanism.removable_bytes` | 机制声称能删掉的字节 |
| `mechanism.removable_ms` | 机制声称能删掉的时间，须附来源 |
| `mechanism.addressable_share_claim` | 机制覆盖的余量比例，∈ [0, 1] |
| `out_of_scope_checks` | 精度/模型/接受率/输出分布，任一为 true 即 `INVALID` |

## 2. 地板：两条硬约束取大

```
t_mem     = bytes / (BW_eff · 1e12) · 1e3
t_compute = flops / (FLOP_eff · 1e12) · 1e3
floor     = max(t_mem, t_compute)
```

`BW_eff = 1.613 TB/s` 是 AutoDL RTX 6000D 主机上算子尺度的实测持续拷贝带宽，不是规格值；`FLOP_eff = 136.8 TFLOPS`
是实测 BF16 稠密 GEMM 峰值。两者都写进 `protocol/gate_rules.yaml` 并由哈希冻结。

**经验地板只抬不降。** 读主导的算子在 300 MB 以下到不了拷贝带宽（实测读上限
0.95–2.57 TB/s），解析地板会凭空造出余量。因此案例可以给出“同样字节的参考 kernel”
耗时，闸门取 `max(analytic, empirical)`。因为地板只升不降，P1（预测 ≥ 实现）按构造安全。

## 3. 可兑现空间：三重约束

```
headroom  = max(0, t_measured - floor)
claimed   = share · headroom
byte_cap  = removable_bytes / (BW_eff · 1e12) · 1e3
applied   = min(claimed, byte_cap, removable_ms, headroom)
```

字节账本是硬约束：一个机制不可能删掉比它实际覆盖的字节更多的时间。
时间账本是第二道：一个延迟主导的机制（例如 0.11 ms 的 selector）只能拿 0.11 ms。
`applied < claimed` 会被记录为 `capped_by_accounting`，而不是悄悄吞掉。

## 4. 区间：拐点必须显式求值

样本 ≥ 5 时用 5–95% 百分位 bootstrap（10000 次重采样，固定种子）；样本更少时退化为
以中位数为中心的参数化区间，且快端用 `med / (1 + z·cv)` 而不是 `med·(1 - z·cv)`，以偏保守。

`speedup(t) = t / (t - applied(t))` 在“余量受限”阶段单调上升，一旦触及 cap 就转为下降。
所以最大值可能落在区间**内部**的拐点 `floor + cap_ms`，而不是任一端点。闸门把落在区间的
所有拐点加入求值集合。本仓库的实测案例里拐点落在区间外侧（`floor + cap` 略高于快端 0.2345 ms，
达到 0.2359 ms），最大值取在端点；但只要拐点落进区间，只算两端就会低估上界。

## 5. 三态判决

| 条件（按顺序） | 判决 |
|---|---|
| 宽松端 < `veto_below_speedup` (1.15×) | `NO-GO` |
| 样本数 < `min_repeats_go` (5) | `INCONCLUSIVE` |
| 保守端 ≥ `go_min_speedup` (1.30×) | 继续检查 |
| ↳ 缺少同字节经验地板 | `INCONCLUSIVE` |
| ↳ 余量占比 < 2% | `INCONCLUSIVE` |
| ↳ CV > 2% | `INCONCLUSIVE` |
| 其余 | `GO` |
| 以上都不满足 | `INCONCLUSIVE` |

**解析地板可以否决，但不可以授予 GO。** 这是 v1.1.0 的核心改动：读主导算子的解析地板
系统性地偏低，只有同字节参考 kernel 才能证明余量真实存在。

## 6. `INVALID` 不是失败，是数据坏了

- `measured_median < floor`（0.1% 容差）：说明账本或参考 kernel 有错。闸门不会帮你下调地板来兜底，因为那样会凭空造出正数余量，把错误变成绿灯。
- 缺少必需字段、未知 dtype、越界比例、机制类不在白名单、精度/模型/接受率/输出分布被改动。
- `rules_sha256` 与当前规则文件不一致。

## 7. 代码指针

- 门槛常数：[gate_rules.yaml](../protocol/gate_rules.yaml)
- 字节/FLOP 账本：[byte_model.py](../gate/byte_model.py)
- 判决逻辑：[decide.py](../gate/decide.py)
- 区间与配对统计：[bootstrap.py](../stats/bootstrap.py)
