/* Static renderer for the Headroom Gate results page.
   Reads ../data/results.json; the schema lives in data/schema.json. */

(() => {
  "use strict";

  const STRINGS = {
    en: {
      "nav.repo": "Repository",
      "nav.spec": "Gate spec",
      "hero.eyebrow": "Kernel research · pre-registered · replayable",
      "hero.title": "Price the ceiling before you write the kernel.",
      "hero.lead": "Most kernel write-ups never state a ceiling, so any gain looks like progress. This project makes the ceiling an input: compulsory bytes, a measured floor, a predicted upper bound. Five archived cases replay offline — four of them say no.",
      "gate.title": "The gate",
      "gate.bytes.title": "Bytes",
      "gate.bytes.body": "Sum the compulsory traffic: weights, activations, output. It cannot be argued down. A mechanism can only remove what it touches.",
      "gate.floor.title": "Floor",
      "gate.floor.body": "max(bytes / BW_eff, flops / FLOP_eff), both measured. A same-bytes reference kernel can raise the floor and can never lower it.",
      "gate.verdict.title": "Verdict",
      "gate.verdict.body": "GO only if the conservative end of the band clears 1.30×. Below 1.15× is NO-GO. In between is INCONCLUSIVE — an honest answer, not a failure.",
      "cases.title": "Five cases, one GO",
      "cases.lead": "Each verdict is arithmetic over recorded samples, so the whole board replays on a laptop with no GPU. The point is not the one win; it is how cheaply the other four ideas die.",
      "cases.th.case": "Case",
      "cases.th.verdict": "Verdict",
      "cases.th.upper": "Predicted ceiling",
      "cases.th.why": "Why",
      "host.title": "On the host: predicted vs realised",
      "host.lead": "The archived case was re-registered on a different GPU with v2 device constants, then measured. The candidate was written only after the ceiling was fixed.",
      "stat.cases": "archived cases",
      "stat.nogo": "NO-GO",
      "stat.go": "GO",
      "stat.e2e": "at model level",
      "stat.e2ePending": "end-to-end pending",
      "sealed.title": "Held-out shapes",
      "sealed.tag": "post-hoc freeze",
      "sealed.lead": "Speedup over a same-bytes control, 95% paired bootstrap. All three shapes are bit-identical to the control.",
      "kpi.predicted": "Predicted ceiling",
      "kpi.predicted.sub": "conservative–generous band",
      "kpi.realised": "Realised",
      "kpi.realised.sub": "Triton kernel, 95% paired CI",
      "kpi.ratio": "Realised / predicted",
      "kpi.ratio.sub": "how much of the priced gain the kernel kept",
      "dispatch.title": "Three ways to move the same bytes",
      "dispatch.lead": "Shipped PyTorch path, a Triton kernel, and a compiled CUDA op. Every arm is bit-identical to the shipped output; only the time differs.",
      "dispatch.th.shape": "Shape",
      "dispatch.th.gain": "shipped → CUDA",
      "dispatch.note": "Triton trails the compiled op at every shape. At b=1 it is 2.2× slower than the CUDA arm — the signature of launch overhead, not bandwidth.",
      "e2e.title": "What survives to the model",
      "e2e.th.workload": "Workload",
      "e2e.th.correct": "Logits",
      "e2e.bitwise": "bit-identical",
      "e2e.headline": "A {op}× operator win arrives as {model}× at model level.",
      "e2e.method": "One merged gate_up GEMM per MLP; only the activation changes. Cache build is outside every window. {correct}",
      "e2e.correct": "All 128 validation logits are bit-identical to the shipped path.",
      "p3.title": "Low-bit pool",
      "p3.pending": "Exploratory timings exist, but no registered selection was run. The frozen policy is published; the pool has no low-bit arms or accuracy trials, so this page makes no low-bit claim.",
      "p3.operators": "Operators in scope",
      "p3.policy": "Policy sha256",
      "limits.title": "What this does not show",
      "foot.generated": "Data snapshot",
      "loading": "Loading measurements…",
      "error": "Could not load data/results.json. Serve the site over HTTP (for example, python3 -m http.server) instead of opening the file directly.",
    },
    zh: {
      "nav.repo": "仓库",
      "nav.spec": "门禁规范",
      "hero.eyebrow": "Kernel研究 · 预注册 · 可复现",
      "hero.title": "先给天花板定价，再写Kernel。",
      "hero.lead": "多数Kernel文章从不给出天花板，于是任何收益都显得像进展。这个项目把天花板变成输入：强制字节量、实测地板、预测上界。五个归档案例可离线重放——其中四个结论是“不要做”。",
      "gate.title": "门禁",
      "gate.bytes.title": "字节",
      "gate.bytes.body": "把强制流量加总：权重、激活、输出。它不能被辩论掉。一个机制只能省掉它真正触及的部分。",
      "gate.floor.title": "地板",
      "gate.floor.body": "max(bytes / BW_eff, flops / FLOP_eff)，两个常数都是实测值。同字节量的参考Kernel可以把地板抬高，绝不会压低。",
      "gate.verdict.title": "结论",
      "gate.verdict.body": "只有区间的保守端超过 1.30× 才是 GO；低于 1.15× 是 NO-GO；中间是 INCONCLUSIVE——这是诚实的答案，不是失败。",
      "cases.title": "五个案例，一个 GO",
      "cases.lead": "每个结论都是对已记录样本的算术，因此整块看板可以在一台没有 GPU 的笔记本上重放。重点不是那一个赢家，而是另外四个想法死得多便宜。",
      "cases.th.case": "案例",
      "cases.th.verdict": "结论",
      "cases.th.upper": "预测天花板",
      "cases.th.why": "原因",
      "host.title": "主机实测：预测与实现",
      "host.lead": "归档案例用 v2 设备常数在另一块 GPU 上重新注册，然后才测量。候选Kernel只在天花板固定之后才写。",
      "stat.cases": "归档案例",
      "stat.nogo": "NO-GO",
      "stat.go": "GO",
      "stat.e2e": "模型级",
      "stat.e2ePending": "端到端待测",
      "sealed.title": "留出形状",
      "sealed.tag": "事后冻结",
      "sealed.lead": "相对同字节量控制组的加速比，95% 配对 bootstrap。三个形状都与控制组逐位一致。",
      "kpi.predicted": "预测天花板",
      "kpi.predicted.sub": "保守端–宽松端区间",
      "kpi.realised": "实测",
      "kpi.realised.sub": "Triton Kernel，95% 配对置信区间",
      "kpi.ratio": "实测 / 预测",
      "kpi.ratio.sub": "定价收益里Kernel真正保住的比例",
      "dispatch.title": "同一批字节，三种走法",
      "dispatch.lead": "PyTorch 现有路径、Triton Kernel、编译版 CUDA 算子。三臂输出逐位一致，差别只在时间。",
      "dispatch.th.shape": "形状",
      "dispatch.th.gain": "shipped → CUDA",
      "dispatch.note": "Triton 在每个形状上都落后于编译版。b=1 时比 CUDA 臂慢 2.2×——这是发射开销的特征，不是带宽。",
      "e2e.title": "到模型层还剩下多少",
      "e2e.th.workload": "负载",
      "e2e.th.correct": "Logits",
      "e2e.bitwise": "逐位一致",
      "e2e.headline": "算子层 {op}× 的收益，到模型层只剩 {model}×。",
      "e2e.method": "每个 MLP 合并成一个 gate_up GEMM，只替换激活。cache 构建不在任何计时窗口内。{correct}",
      "e2e.correct": "128 个验证 logits 与现有路径逐位一致。",
      "p3.title": "低比特池",
      "p3.pending": "已有探索性计时，但没有运行注册过的选择。冻结策略已发布；池里没有低比特臂，也没有精度 trial，因此本页不做任何低比特声明。",
      "p3.operators": "范围内算子",
      "p3.policy": "策略 sha256",
      "limits.title": "这页不证明什么",
      "foot.generated": "数据快照",
      "loading": "正在载入测量数据…",
      "error": "无法加载 data/results.json。请用 HTTP 打开本站（例如 python3 -m http.server），不要直接双击文件。",
    },
  };

  const SVG_NS = "http://www.w3.org/2000/svg";
  let lang = "en";
  let data = null;

  try {
    const saved = localStorage.getItem("hgk.lang");
    if (saved === "zh" || saved === "en") lang = saved;
  } catch (_) { /* storage can be blocked */ }

  const t = (key, vars) => {
    let s = (STRINGS[lang] && STRINGS[lang][key]) || STRINGS.en[key] || key;
    if (vars) for (const [k, v] of Object.entries(vars)) s = s.replace(`{${k}}`, v);
    return s;
  };

  const num = (x, d = 2) => {
    if (x === null || x === undefined || Number.isNaN(x)) return "—";
    return Number(x).toFixed(d);
  };

  const pick = (obj, base) => {
    if (!obj) return "";
    if (lang === "zh" && obj[`${base}_zh`]) return obj[`${base}_zh`];
    return obj[base] || "";
  };

  function el(tag, attrs = {}, text) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") node.className = v;
      else if (k === "html") node.innerHTML = v;
      else if (v !== null && v !== undefined) node.setAttribute(k, v);
    }
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  function svgEl(tag, attrs = {}, text) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v !== null && v !== undefined) node.setAttribute(k, v);
    }
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  function applyStatic() {
    document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
    document.querySelectorAll("[data-i18n]").forEach((node) => {
      const key = node.getAttribute("data-i18n");
      const text = t(key);
      if (text) node.textContent = text;
    });
    const btn = document.getElementById("lang");
    btn.textContent = lang === "en" ? "中文" : "EN";
    btn.setAttribute("aria-label", lang === "en" ? "切换到中文" : "Switch to English");
    const spec = document.getElementById("spec-link");
    if (spec) {
      spec.href = lang === "zh"
        ? "https://github.com/SanyHe/headroom-gate-kernel-rsi/blob/main/docs/01-gate.md"
        : "https://github.com/SanyHe/headroom-gate-kernel-rsi/blob/main/docs/en/01-gate.md";
    }
  }

  function renderStats() {
    const host = document.getElementById("statstrip");
    host.replaceChildren();
    const s = data.summary || {};
    const measured = data.e2e && data.e2e.status === "measured";
    const model = measured
      ? `${num(data.e2e.workloads[0].cuda.speedup, 2)}×`
      : "—";
    const items = [
      [s.cases, t("stat.cases"), ""],
      [s.no_go, t("stat.nogo"), "no"],
      [s.go, t("stat.go"), "go"],
      [model, measured ? t("stat.e2e") : t("stat.e2ePending"), measured ? "go" : ""],
    ];
    for (const [value, label, cls] of items) {
      const box = el("div", { class: `stat ${cls}`.trim() });
      box.append(el("b", {}, String(value)));
      box.append(el("span", {}, label));
      host.append(box);
    }
  }

  function renderCases() {
    const tbody = document.querySelector("#case-table tbody");
    tbody.replaceChildren();
    for (const c of data.cases || []) {
      const tr = el("tr");
      const idCell = el("td");
      idCell.append(el("span", { class: "cid" }, c.id));
      tr.append(idCell);
      const verdict = el("td");
      verdict.append(el("span", {
        class: `pill ${c.verdict === "GO" ? "go" : c.verdict === "NO-GO" ? "nogo" : "inc"}`,
      }, c.verdict));
      tr.append(verdict);
      const range = Array.isArray(c.upper_range)
        ? `${num(c.upper_range[0], c.upper_range[1] < 1.01 ? 4 : 2)}–${num(c.upper_range[1], c.upper_range[1] < 1.01 ? 4 : 2)}×`
        : `${num(c.upper_point, 2)}×`;
      tr.append(el("td", { class: "num" }, `${range}`));
      tr.append(el("td", { class: "why" }, pick(c, "reason")));
      tbody.append(tr);
    }
  }

  function renderHost() {
    const host = document.getElementById("host-kpis");
    host.replaceChildren();
    const run = data.host_run || {};
    const cards = [
      [t("kpi.predicted"), `${num(run.predicted.point, 2)}×`,
        `${num(run.predicted.min, 3)}–${num(run.predicted.max, 3)}× · ${t("kpi.predicted.sub")}`, ""],
      [t("kpi.realised"), `${num(run.realised.speedup, 2)}×`,
        `${num(run.realised.ci95[0], 3)}–${num(run.realised.ci95[1], 3)} · ${t("kpi.realised.sub")}`, "go"],
      [t("kpi.ratio"), num(run.realised_vs_predicted, 2),
        t("kpi.ratio.sub"), "warn"],
    ];
    for (const [k, v, s, cls] of cards) {
      const card = el("div", { class: `kpi ${cls}`.trim() });
      card.append(el("span", { class: "k" }, k));
      card.append(el("span", { class: "v" }, v));
      card.append(el("span", { class: "s" }, s));
      host.append(card);
    }

    const note = document.getElementById("sealed-note");
    note.textContent = pick(data.sealed, "note");

    const cross = document.getElementById("crosscheck");
    const shapes = (data.crosscheck && data.crosscheck.shapes) || [];
    const gains = shapes.map((x) => `${num(x.speedup, 2)}×`).join(" / ");
    cross.textContent = lang === "zh"
      ? `v2 常数 ${num(data.host.hbm_realistic_tbs, 3)} TB/s 在 ${shapes.length} 个形状上复核：${gains}，全部逐位一致。`
      : `The v2 constant ${num(data.host.hbm_realistic_tbs, 3)} TB/s was exercised at ${shapes.length} shapes: ${gains}, all bit-identical.`;
  }

  function sealedChart() {
    const wrap = document.getElementById("sealed-chart");
    wrap.replaceChildren();
    const rows = data.sealed.holdouts || [];
    const W = 720, rowH = 58, top = 26, left = 128, right = 132;
    const H = top + rows.length * rowH + 26;
    const x0 = 1.0, x1 = 2.0;
    const plotW = W - left - right;
    const x = (v) => left + ((v - x0) / (x1 - x0)) * plotW;
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
      "aria-label": "Held-out shapes: speedup with 95% confidence intervals" });

    for (let tick = 1.0; tick <= 2.001; tick += 0.25) {
      const gx = x(tick);
      svg.append(svgEl("line", { class: "gridline", x1: gx, y1: top - 10, x2: gx, y2: H - 24 }));
      svg.append(svgEl("text", { class: "axis", x: gx, y: H - 8, "text-anchor": "middle" },
        `${tick.toFixed(2)}×`));
    }

    rows.forEach((r, i) => {
      const y = top + i * rowH + rowH / 2;
      svg.append(svgEl("text", { class: "label", x: 0, y: y + 4 }, r.id));
      svg.append(svgEl("rect", {
        class: "bar-sealed", x: x(x0), y: y - 9,
        width: Math.max(2, x(r.speedup) - x(x0)), height: 18, rx: 3,
      }));
      svg.append(svgEl("line", { class: "ci-line", x1: x(r.ci95[0]), y1: y, x2: x(r.ci95[1]), y2: y }));
      for (const end of r.ci95) {
        svg.append(svgEl("line", { class: "ci-cap", x1: x(end), y1: y - 5, x2: x(end), y2: y + 5 }));
      }
      svg.append(svgEl("text", { class: "value", x: x(r.ci95[1]) + 10, y: y + 4 },
        `${num(r.speedup, 2)}×`));
      svg.append(svgEl("text", { class: "axis", x: W - 44, y: y + 4 },
        `${num(r.candidate_ms, 3)} ms`));
    });
    wrap.append(svg);
  }

  function dispatchChart() {
    const wrap = document.getElementById("dispatch-chart");
    wrap.replaceChildren();
    const rows = data.dispatch.shapes || [];
    const W = 760, rowH = 84, top = 40, left = 178, right = 74;
    const H = top + rows.length * rowH + 22;
    const plotW = W - left - right;
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
      "aria-label": "Three-arm operator timing by shape" });

    const legend = [["shipped", "bar-shipped"], ["Triton", "bar-triton"], ["CUDA", "bar-cuda"]];
    let lx = left;
    for (const [name, cls] of legend) {
      svg.append(svgEl("rect", { class: cls, x: lx, y: 8, width: 12, height: 12, rx: 2 }));
      svg.append(svgEl("text", { class: "axis", x: lx + 18, y: 18 }, name));
      lx += 18 + name.length * 7 + 22;
    }

    rows.forEach((r, i) => {
      const baseY = top + i * rowH;
      svg.append(svgEl("text", { class: "label", x: 0, y: baseY + 22 }, r.id));
      svg.append(svgEl("text", { class: "axis", x: 0, y: baseY + 38 }, r.role));
      const max = Math.max(r.shipped_ms, r.triton_ms, r.cuda_ms);
      const arms = [
        [r.shipped_ms, "bar-shipped"],
        [r.triton_ms, "bar-triton"],
        [r.cuda_ms, "bar-cuda"],
      ];
      arms.forEach(([ms, cls], k) => {
        const y = baseY + 6 + k * 18;
        const w = Math.max(2, (ms / max) * plotW);
        svg.append(svgEl("rect", { class: cls, x: left, y, width: w, height: 13, rx: 2 }));
        svg.append(svgEl("text", { class: "value", x: left + w + 6, y: y + 11 },
          `${ms.toFixed(4)} ms`));
      });
    });
    wrap.append(svg);
  }

  function e2eChart() {
    const wrap = document.getElementById("e2e-chart");
    wrap.replaceChildren();
    const rows = (data.e2e && data.e2e.workloads) || [];
    const W = 760, rowH = 74, top = 34, left = 190, right = 92;
    const H = top + rows.length * rowH + 22;
    const x0 = 0.9, x1 = 1.1;
    const plotW = W - left - right;
    const x = (v) => left + ((v - x0) / (x1 - x0)) * plotW;
    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
      "aria-label": "End-to-end speedup around parity" });

    for (const tick of [0.9, 0.95, 1.0, 1.05, 1.1]) {
      const gx = x(tick);
      svg.append(svgEl("line", {
        class: tick === 1.0 ? "baseline" : "gridline",
        x1: gx, y1: top - 10, x2: gx, y2: H - 22,
      }));
      svg.append(svgEl("text", { class: tick === 1.0 ? "baseline-label" : "axis",
        x: gx, y: H - 6, "text-anchor": "middle" }, `${tick.toFixed(2)}×`));
    }

    rows.forEach((r, i) => {
      const baseY = top + i * rowH;
      svg.append(svgEl("text", { class: "label", x: 0, y: baseY + 22 }, r.label));
      const arms = [
        [r.triton.speedup, "bar-triton", "Triton"],
        [r.cuda.speedup, "bar-cuda", "CUDA"],
      ];
      arms.forEach(([v, cls, name], k) => {
        const y = baseY + 6 + k * 22;
        const start = Math.min(x(1.0), x(v));
        const width = Math.max(1.5, Math.abs(x(v) - x(1.0)));
        svg.append(svgEl("rect", { class: cls, x: start, y, width, height: 15, rx: 2 }));
        svg.append(svgEl("text", { class: "value", x: left + plotW + 8, y: y + 12 },
          `${name} ${num(v, 3)}×`));
      });
    });
    wrap.append(svg);
  }

  function renderDispatchTable() {
    const tbody = document.querySelector("#dispatch-table tbody");
    tbody.replaceChildren();
    for (const r of data.dispatch.shapes || []) {
      const tr = el("tr");
      const shape = el("td");
      shape.append(el("span", { class: "cid" }, r.id));
      shape.append(document.createElement("br"));
      shape.append(el("span", {}, r.role));
      tr.append(shape);
      tr.append(el("td", { class: "num" }, `${r.shipped_ms.toFixed(4)} ms`));
      tr.append(el("td", { class: "num" }, `${r.triton_ms.toFixed(4)} ms`));
      tr.append(el("td", { class: "num" }, `${r.cuda_ms.toFixed(4)} ms`));
      tr.append(el("td", { class: "num" }, `${num(r.shipped_vs_cuda, 3)}×  [${num(r.ci95[0], 2)}, ${num(r.ci95[1], 2)}]`));
      tbody.append(tr);
    }
  }

  function renderE2E() {
    const e = data.e2e;
    const head = document.getElementById("e2e-headline");
    const method = document.getElementById("e2e-method");
    const boundary = document.getElementById("e2e-boundary");
    const table = document.querySelector("#e2e-table tbody");

    if (!e || e.status !== "measured") {
      head.textContent = "—";
      method.textContent = t("stat.e2ePending");
      boundary.textContent = "";
      table.replaceChildren();
      return;
    }

    const op = e.operator_speedup.cuda;
    const modelLevel = e.workloads[0].cuda.speedup;
    head.innerHTML = "";
    const text = t("e2e.headline", { op: num(op, 2), model: num(modelLevel, 2) });
    // Split the sentence around the two numbers so they can carry color.
    const opStr = `${num(op, 2)}×`;
    const modelStr = `${num(modelLevel, 2)}×`;
    const [before, rest] = text.split(opStr);
    const [middle, after] = (rest || "").split(modelStr);
    head.append(document.createTextNode(before || ""));
    head.append(el("span", { class: "hot" }, opStr));
    head.append(el("span", { class: "arrow" }, "→"));
    head.append(document.createTextNode(middle || ""));
    head.append(el("span", { class: "cold" }, modelStr));
    head.append(document.createTextNode(after || ""));

    method.textContent = t("e2e.method", {
      correct: e.correctness.logits_bitwise_equal ? t("e2e.correct") : "",
    });
    boundary.textContent = pick(e, "boundary");

    table.replaceChildren();
    for (const w of e.workloads) {
      const tr = el("tr");
      tr.append(el("td", { class: "cid" }, w.label));
      tr.append(el("td", { class: "num" }, `${num(w.strided_ms, 1)} ms`));
      tr.append(el("td", { class: "num" },
        `${num(w.triton.fused_ms, 1)} ms · ${num(w.triton.speedup, 3)}×`));
      tr.append(el("td", { class: "num" },
        `${num(w.cuda.fused_ms, 1)} ms · ${num(w.cuda.speedup, 3)}×`));
      tr.append(el("td", {}, e.correctness.logits_bitwise_equal ? t("e2e.bitwise") : "—"));
      table.append(tr);
    }
  }

  function renderP3() {
    const box = document.getElementById("p3-panel");
    box.replaceChildren();
    const p = data.p3 || {};
    if (p.status !== "measured") {
      box.append(el("p", {}, t("p3.pending")));
      const row = el("div", { class: "row" });
      row.append(el("span", { class: "k" }, `${t("p3.operators")}:`));
      for (const op of p.operators || []) row.append(el("span", { class: "tag" }, op));
      box.append(row);
      box.append(el("p", { class: "hash" }, `${t("p3.policy")} ${p.policy_sha256 || "—"}`));
      return;
    }
    box.append(el("p", {}, pick(p, "note")));
    const row = el("div", { class: "row" });
    for (const op of p.operators || []) row.append(el("span", { class: "tag" }, op));
    box.append(row);
  }

  function renderLimits() {
    const list = document.getElementById("limits-list");
    list.replaceChildren();
    for (const c of data.caveats || []) {
      list.append(el("li", {}, lang === "zh" ? (c.zh || c.en) : c.en));
    }
    document.getElementById("generated").textContent =
      `${t("foot.generated")} ${data.generated_at.replace("T", " ").slice(0, 16)}`;
  }

  function renderAll() {
    if (!data) return;
    renderStats();
    renderCases();
    renderHost();
    sealedChart();
    dispatchChart();
    renderDispatchTable();
    renderE2E();
    e2eChart();
    renderP3();
    renderLimits();
  }

  function setLang(next) {
    lang = next;
    try { localStorage.setItem("hgk.lang", lang); } catch (_) { /* ignore */ }
    applyStatic();
    renderAll();
  }

  async function boot() {
    applyStatic();
    document.getElementById("lang").addEventListener("click", () => {
      setLang(lang === "en" ? "zh" : "en");
    });
    try {
      const res = await fetch("data/results.json", { cache: "no-store" });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      data = await res.json();
      renderAll();
    } catch (err) {
      const main = document.getElementById("main");
      const box = el("div", { class: "error" }, t("error"));
      box.append(document.createElement("br"));
      box.append(el("span", { class: "mono" }, String(err.message || err)));
      main.replaceChildren(box);
      console.error(err);
    }
  }

  boot();
})();
