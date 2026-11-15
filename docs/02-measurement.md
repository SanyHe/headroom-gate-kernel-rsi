# 02 — 测量协议与后台运行契约

判决的质量上限由测量决定。这一节规定两件事：怎么计时，以及怎么让一次长时间运行
在断线、关终端、SSH 掉线之后仍然可判定。

## 1. 计时协议（所有脚本共用同一语义）

- **CUDA events 包住单次 launch**，不是 `time.time()`，也不是整段循环的平均。
- **每次重复前刷 L2**（读一遍 256 MB 缓冲），否则第二个算子会命中小算子的热缓存。
- **warmup 与 repeats 分开记录**；repo 默认 10 warmup / 20 repeats，单样本案例另注。
- **时钟只记录、不调节**（`clocks: recorded-only`），避免把锁频当成算法收益。
- **噪声地板 1%**：低于 1% 的差异在本 harness 上记为“不可测量”，两个方向都不许当成收益。
- **原始样本全部保留**：`measured_ms` 是列表，不是中位数。区间、CV、P1 都从样本重算。

证据文件 (`cases/<id>/evidence/*.json`) 必须能回答：测了什么形状、什么 dtype、多少
warmup/repeats、用什么计时、中位数多少、参考 kernel 是多少。`case.yaml` 的
`measurement_source` 指向具体文件与字段。

## 2. 为什么 `case.yaml` 只追加

预注册的意义在于：写下假设之后再看到数字，就不能改假设。所以

- `measured_ms`、`removable_*`、`addressable_share_claim` 在判决之后不得改写；
- 事后认识写进 `post_measurement_note`（attn、gemv 两个案例都保留了被证伪的原始假设）；
- 新判决追加到 `ledger.jsonl`，旧判决留在原处作为历史。

## 3. 后台运行契约

长时间测量不能依赖终端还活着。`scripts/` 实现的契约是：

```
QUEUED ──► RUNNING ──► SUCCEEDED | FAILED | CANCELLED
```

| 脚本 | 作用 |
|---|---|
| `scripts/detach.sh <job> <cmd...>` | `setsid` 新会话启动 supervisor，`</dev/null` 防阻塞，`set -m` 保证 SIGINT 可捕获；打印 run 目录 |
| `scripts/job_supervisor.sh` | 启动子进程、转发信号、写 `status.json`，每 5 秒心跳 |
| `scripts/check_run.sh <run_dir>` | 打印 status + 心跳年龄 + 日志尾部；退出码 0/1/2 对应 成功/失败/仍在跑 |
| `scripts/stop_run.sh <run_dir>` | 先 SIGINT，等 30 秒；再 SIGTERM，等 10 秒；**绝不 kill -9** |
| `scripts/resume.sh <run_dir>` | 从 `command.json` 重放 argv，`resume_count + 1`；拒绝在 RUNNING/QUEUED 上重入 |
| `scripts/lib/runmeta.py` | `init / status / heartbeat / checkpoint`，所有 JSON 原子写 |

`status.json` 是唯一真相：成功不由“终端看起来结束了”决定，而由 `state` 与 `exit_code` 决定。

一次 run 目录里应当有：`command.txt`、`command.json`、`environment.json`（主机、Python、
git commit、可见 GPU）、`input_manifest.json`（每个输入文件的 sha256）、`status.json`、
`heartbeat.json`、`checkpoint.json`、`stdout.log`、`stderr.log`、`artifacts/`。

## 4. 平台差异（已实测）

- Linux 上 `setsid` 给出独立会话，`ps -o sid=` 有效。
- macOS 没有 `setsid`，脚本退化为 `nohup`，`ps -o sid` 不可用（BSD 只有 `sess`），
  因此 supervisor 会校验 sid 是否数字，否则记为 0。本地只用于契约冒烟测试。
- 朴素 `cmd &` 会让子进程继承“忽略 SIGINT”，导致优雅停止永远不生效。supervisor 内部
  用 `set -m` 启动子进程，`stop_run.sh` 的 SIGINT 才会真正到达算子。

## 5. 当前边界

本仓库尚未落地 GPU 侧脚本（`measure_baseline.py`、`measure_candidate.py`、
`sealed_eval.py`、`kernel.py`、`measure_dispatch.py`、`e2e_qwen3_8b.py`、`analyze_e2e.py`、
`scripts/compare.py`、`p3/measure_pool.py`）。它们全部标记为 `blocked`，不伪造成“已完成”。
现有五个归档判决因此是从已记录样本复算，而不是在本机重新测量。
