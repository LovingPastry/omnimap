"""Summarize the grad_eps sweep + Fisher contribution diagnostics.

    python sim/ablation/analyze_eps_contrib.py sim/sim_outputs/eps_contrib

Layout: <root>/eps<E>/<start>/r<k>/{loop_log.jsonl, quality.jsonl, summary.json, contrib_diag.jsonl}.
Writes <root>/report/{metrics.json, tables.md, *.png}.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

COVER_DEG = 25.0  # same neighbourhood as the stop monitor's view coverage
RNG = np.random.default_rng(0)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _quality_at(rows: list[dict], step: int, key: str) -> float:
    best = None
    for row in rows:
        if row["idx"] <= step:
            best = row
    return float(best[key]) if best is not None else float("nan")


def _turn_angles_deg(log: list[dict]) -> np.ndarray:
    """Angle between consecutive commanded tangent velocity directions (world frame)."""
    vel = np.array([r.get("velocity_world", [np.nan] * 3) for r in log], dtype=float)
    norms = np.linalg.norm(vel, axis=1)
    ok = norms > 1e-9
    unit = np.where(ok[:, None], vel / np.maximum(norms, 1e-12)[:, None], np.nan)
    cos = np.sum(unit[1:] * unit[:-1], axis=1)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def run_metrics(run_dir: Path, budget: int) -> dict:
    log = _load_jsonl(run_dir / "loop_log.jsonl")
    quality = _load_jsonl(run_dir / "quality.jsonl")
    summary = json.loads((run_dir / "summary.json").read_text())
    phi = np.degrees([r["current_phi"] for r in log])
    grid = np.arange(0, budget, 10)
    curve = np.array([_quality_at(quality, int(s), "heldout_psnr") for s in grid])
    last = log[-1]
    turns = _turn_angles_deg(log)
    return {
        "steps_run": summary["steps_run"],
        "path_m": summary["path_length_m"],
        "final_psnr": quality[-1]["heldout_psnr"],
        "final_psnr_min_view": quality[-1]["heldout_psnr_min"],
        "final_completeness": quality[-1]["completeness"],
        "psnr_auc": float(np.nanmean(curve)),
        "psnr_curve": curve.tolist(),
        "phi_min_deg": float(np.min(phi)),
        "phi_max_deg": float(np.max(phi)),
        "view_coverage": float(last.get("nbv_stop_view_coverage", np.nan)),
        "lon_bins": int(last.get("longitude_bins_visited", 0)),
        "speed_median_mps": float(np.nanmedian([r.get("linear_speed_applied", np.nan) for r in log])),
        "turn_median_deg": float(np.nanmedian(turns)) if turns.size else float("nan"),
        "turn_gt90_frac": float(np.nanmean(turns > 90.0)) if turns.size else float("nan"),
    }


def mean_std(values) -> tuple[float, float]:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0


def boot_diff_ci(a, b, n: int = 10000) -> tuple[float, float, float]:
    """Mean(a) - mean(b) with a 95% bootstrap CI (independent resampling)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = [RNG.choice(a, a.size).mean() - RNG.choice(b, b.size).mean() for _ in range(n)]
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def fmt(m: float, s: float, digits: int = 2) -> str:
    return f"{m:.{digits}f} ± {s:.{digits}f}"


# ---------------------------------------------------------------------------
# Contribution diagnostics
# ---------------------------------------------------------------------------

UNOBS = ("score_h_lt_0.1L", "score_h_0.1L_L")  # history < lambda: denominator dominated by lambda
LAMS = ("score_lam0.1", "score_lam1", "score_lam10", "score_lam100")


def _unobs_share(row: dict) -> float:
    s = row["score"]
    return sum(row[k] for k in UNOBS) / s if s > 0 else float("nan")


def _observed(row: dict) -> float:
    return row["score"] - sum(row[k] for k in UNOBS)


def contrib_rows(run_dir: Path) -> list[dict]:
    """One row per diagnostic snapshot with the quantities the report needs."""
    out = []
    for snap in _load_jsonl(run_dir / "contrib_diag.jsonl"):
        cur = snap["current_sphere"]
        unc = [p for p in snap["probes"] if p["nearest_visited_deg"] > COVER_DEG]
        cov = [p for p in snap["probes"] if p["nearest_visited_deg"] <= COVER_DEG]
        row = {
            "idx": snap["idx"],
            "num_keyframes": snap["num_keyframes"],
            "elem_frac_h_lt_L": snap["elem_frac_h_lt_L"],
            "n_uncovered": len(unc),
            "cur_unobs_share": _unobs_share(cur),
            "unc_unobs_share": float(np.mean([_unobs_share(p) for p in unc])) if unc else float("nan"),
            "cov_unobs_share": float(np.mean([_unobs_share(p) for p in cov])) if cov else float("nan"),
        }
        peak = snap["stop_max_window_mean"]
        if unc:
            best = max(unc, key=lambda p: p["score"])
            row["ratio_vs_peak"] = best["score"] / peak if peak and peak > 0 and math.isfinite(peak) else float("nan")
            row["best_unc_unobs_share"] = _unobs_share(best)
            for k in LAMS:
                # instantaneous: best uncovered probe vs the current direction, same lambda'
                c = cur[k]
                row[f"ratio_now_{k[6:]}"] = max(p[k] for p in unc) / c if c > 0 else float("nan")
            row["ratio_now_observed_only"] = (
                max(_observed(p) for p in unc) / max(_observed(cur), 1e-12)
            )
        if unc and cov:
            # The current pose is (nearly) a keyframe, so every element it sees has
            # cur ~ history and the current score is a local minimum by construction.
            # Off-keyframe vs off-keyframe is the fairer "is there more out there?".
            for k in LAMS:
                row[f"ratio_uc_{k[6:]}"] = max(p[k] for p in unc) / max(max(p[k] for p in cov), 1e-12)
            row["ratio_uc_observed_only"] = max(_observed(p) for p in unc) / max(
                max(_observed(p) for p in cov), 1e-12
            )
        out.append(row)
    return out


# ---------------------------------------------------------------------------


def main(root: Path) -> None:
    report = root / "report"
    report.mkdir(parents=True, exist_ok=True)
    budget = 300
    runs: dict[str, list[dict]] = defaultdict(list)
    contrib: dict[str, list[list[dict]]] = defaultdict(list)
    for eps_dir in sorted(root.glob("eps*")):
        eps = eps_dir.name[3:]
        for run_dir in sorted(eps_dir.glob("*/r*")):
            if not (run_dir / "summary.json").exists():
                continue
            m = run_metrics(run_dir, budget)
            m["start"], m["rep"] = run_dir.parent.name, run_dir.name
            runs[eps].append(m)
            contrib[eps].append(contrib_rows(run_dir))
    eps_keys = sorted(runs, key=float)
    lines: list[str] = []

    # --- Experiment 1 table -------------------------------------------------
    lines.append("## 实验 1：grad_eps 扫描（每档 n = 起点数 × 重复数）\n")
    cols = [
        ("final_psnr", "终态 PSNR (dB)", 2),
        ("psnr_auc", "PSNR 曲线均值 (dB)", 2),
        ("view_coverage", "视角覆盖", 3),
        ("lon_bins", "经度分区 /18", 1),
        ("path_m", "路径 (m)", 2),
        ("phi_min_deg", "最低仰角 (°)", 1),
        ("phi_max_deg", "最高仰角 (°)", 1),
        ("turn_median_deg", "相邻指令转角中位数 (°)", 1),
        ("turn_gt90_frac", "转角>90° 占比", 3),
    ]
    lines.append("| eps | n | " + " | ".join(c[1] for c in cols) + " |")
    lines.append("|---" * (len(cols) + 2) + "|")
    for e in eps_keys:
        cells = [fmt(*mean_std([r[k] for r in runs[e]]), d) for k, _, d in cols]
        lines.append(f"| {e} | {len(runs[e])} | " + " | ".join(cells) + " |")

    lines.append("\n逐起点终态 PSNR / 视角覆盖（重复取均值）：\n")
    starts = sorted({r["start"] for e in eps_keys for r in runs[e]})
    lines.append("| eps | " + " | ".join(f"{s} PSNR | {s} 覆盖" for s in starts) + " |")
    lines.append("|---" * (2 * len(starts) + 1) + "|")
    for e in eps_keys:
        cells = []
        for s in starts:
            sub = [r for r in runs[e] if r["start"] == s]
            cells.append(fmt(*mean_std([r["final_psnr"] for r in sub])))
            cells.append(fmt(*mean_std([r["view_coverage"] for r in sub]), 3))
        lines.append(f"| {e} | " + " | ".join(cells) + " |")

    base = eps_keys[0]
    lines.append(f"\n相对 eps={base} 的差（均值差，95% bootstrap CI）：\n")
    lines.append("| eps | 终态 PSNR | PSNR 曲线均值 | 视角覆盖 |")
    lines.append("|---|---|---|---|")
    for e in eps_keys[1:]:
        cells = []
        for k, d in (("final_psnr", 2), ("psnr_auc", 2), ("view_coverage", 3)):
            m, lo, hi = boot_diff_ci([r[k] for r in runs[e]], [r[k] for r in runs[base]])
            cells.append(f"{m:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]")
        lines.append(f"| {e} | " + " | ".join(cells) + " |")

    # --- Experiment 2 table -------------------------------------------------
    lines.append("\n## 实验 2：Fisher 分数的来源（history < λ 的参数元素 = 未观测）\n")
    lines.append(
        "| eps | 快照数 | 未观测元素占比 | 当前视角分数中未观测占比 | 已覆盖方向中未观测占比 | "
        "未覆盖方向中未观测占比 | 最佳未覆盖方向中未观测占比 | 全局比（对历史峰值，同 stop 判据） |"
    )
    lines.append("|---" * 8 + "|")
    pooled: list[dict] = []
    for e in eps_keys:
        rows = [r for run in contrib[e] for r in run if r["idx"] > 0]
        pooled += rows
        vals = lambda k: mean_std([r.get(k, np.nan) for r in rows])  # noqa: E731
        lines.append(
            f"| {e} | {len(rows)} | {fmt(*vals('elem_frac_h_lt_L'), 3)} | {fmt(*vals('cur_unobs_share'), 3)} | "
            f"{fmt(*vals('cov_unobs_share'), 3)} | {fmt(*vals('unc_unobs_share'), 3)} | "
            f"{fmt(*vals('best_unc_unobs_share'), 3)} | {fmt(*vals('ratio_vs_peak'), 2)} |"
        )

    for prefix, title in (
        ("ratio_now_", "最佳未覆盖方向 / 当前方向"),
        ("ratio_uc_", "最佳未覆盖方向 / 最佳已覆盖方向（都不在关键帧上）"),
    ):
        lines.append(f"\n反事实：把分母换成 history + λ'，{title} 的比值（全部快照合并，中位数 [四分位]）：\n")
        lines.append("| 阶段 | λ'=0.1（实际） | λ'=1 | λ'=10 | λ'=100 | 只计已观测元素 |")
        lines.append("|---|---|---|---|---|---|")
        for lo_step, hi_step, label in ((1, 100, "步 30–90"), (100, 200, "步 120–180"), (200, 301, "步 210–270"), (1, 301, "全部")):
            sub = [r for r in pooled if lo_step <= r["idx"] < hi_step and f"{prefix}lam0.1" in r]
            cells = []
            for k in ("lam0.1", "lam1", "lam10", "lam100", "observed_only"):
                v = np.asarray([r[prefix + k] for r in sub if np.isfinite(r[prefix + k])])
                cells.append(f"{np.median(v):.2f} [{np.percentile(v, 25):.2f}, {np.percentile(v, 75):.2f}]" if v.size else "–")
            lines.append(f"| {label} (n={len(sub)}) | " + " | ".join(cells) + " |")

    (report / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (report / "metrics.json").write_text(
        json.dumps({"runs": runs, "contrib": contrib}, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print("\n".join(lines))
    _plots(report, runs, contrib, eps_keys, budget)


def _plots(report: Path, runs, contrib, eps_keys, budget) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"skip plots: {exc}")
        return
    colors = ["#4C72B0", "#55A868", "#DD8452", "#C44E52", "#8172B3"]
    grid = np.arange(0, budget, 10)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, e in enumerate(eps_keys):
        curves = np.array([r["psnr_curve"] for r in runs[e]])
        m = np.nanmean(curves, axis=0)
        s = np.nanstd(curves, axis=0)
        axes[0].plot(grid, m, color=colors[i % 5], label=f"eps={e}")
        axes[0].fill_between(grid, m - s, m + s, color=colors[i % 5], alpha=0.15)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("held-out PSNR (dB)")
    axes[0].legend()
    axes[0].set_title("PSNR vs step (mean ± sd)")
    for i, e in enumerate(eps_keys):
        x = np.full(len(runs[e]), float(e)) * (1 + 0.04 * (RNG.random(len(runs[e])) - 0.5))
        axes[1].scatter(x, [r["view_coverage"] for r in runs[e]], color=colors[i % 5])
        axes[2].scatter(x, [r["final_psnr"] for r in runs[e]], color=colors[i % 5])
    for ax, lab in ((axes[1], "view coverage"), (axes[2], "final PSNR (dB)")):
        ax.set_xscale("log")
        ax.set_xlabel("grad_eps (rad)")
        ax.set_ylabel(lab)
    fig.tight_layout()
    fig.savefig(report / "eps_sweep.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    pooled = [r for e in eps_keys for run in contrib[e] for r in run if r["idx"] > 0]
    idx = np.array([r["idx"] for r in pooled])
    for key, lab, c in (
        ("cur_unobs_share", "current view", colors[0]),
        ("unc_unobs_share", "uncovered probes", colors[3]),
        ("cov_unobs_share", "covered probes", colors[1]),
    ):
        v = np.array([r[key] for r in pooled])
        steps = sorted(set(idx))
        axes[0].plot(steps, [np.nanmean(v[idx == s]) for s in steps], marker="o", color=c, label=lab)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("share of score from history < λ")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    for j, (key, lab) in enumerate(
        (("ratio_now_lam0.1", "λ'=0.1 (actual)"), ("ratio_now_lam1", "λ'=1"), ("ratio_now_lam10", "λ'=10"),
         ("ratio_now_lam100", "λ'=100"), ("ratio_now_observed_only", "observed only"))
    ):
        sub = [r for r in pooled if key in r]
        steps = sorted({r["idx"] for r in sub})
        med = [np.nanmedian([r[key] for r in sub if r["idx"] == s]) for s in steps]
        axes[1].plot(steps, med, marker="o", color=colors[j % 5], label=lab)
    axes[1].axhline(1.0, color="gray", lw=0.8, ls="--")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("best uncovered / current (median)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(report / "contrib_diag.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "sim/sim_outputs/eps_contrib"))
