"""Summarize an nbv_policy ablation run directory into tables and plots.

    python sim/ablation/analyze_nbv_ablation.py sim/sim_outputs/nbv_ablation

Layout expected: <root>/<scene>/<arm>/<start>/{loop_log.jsonl, quality.jsonl, summary.json}.
Writes <root>/report/{metrics.json, tables.md, *.png}.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ARM_ORDER = [
    "legacy",
    "tangent_only",
    "full",
    "full_no_tangent",
    "full_no_feas",
    "full_no_mask",
    "full_no_rate",
    "full_no_stop",
]
RIM_DEG = 15.0  # feasibility band lower limit used by the arms that enable it


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def quality_at(rows: list[dict], step: int, key: str) -> float:
    """Quality at `step`; a run that stopped earlier keeps its last value (the map stops improving)."""
    best = None
    for row in rows:
        if row["idx"] <= step:
            best = row
    return float(best[key]) if best is not None else float("nan")


def run_metrics(run_dir: Path, budget: int) -> dict:
    log = _load_jsonl(run_dir / "loop_log.jsonl")
    quality = _load_jsonl(run_dir / "quality.jsonl")
    summary = json.loads((run_dir / "summary.json").read_text())
    phi = np.degrees([r["current_phi"] for r in log]) if log else np.array([np.nan])
    lin_raw = np.array([r.get("linear_speed_raw", np.nan) for r in log], dtype=float)
    lin = np.array([r.get("linear_speed_applied", np.nan) for r in log], dtype=float)
    vmax = np.array([r.get("linear_speed_limit", np.nan) for r in log], dtype=float)
    moving = [r for r in log if not r.get("should_stop")]
    last = log[-1] if log else {}
    steps = [r["idx"] for r in quality]
    psnr = [r["heldout_psnr"] for r in quality]
    # PSNR area under curve over the common step budget, normalized per step.
    grid = np.arange(0, budget, 10)
    curve = np.array([quality_at(quality, int(s), "heldout_psnr") for s in grid])
    return {
        "steps_run": summary["steps_run"],
        "stop_reason": summary["stop_reason"],
        "path_m": summary["path_length_m"],
        "final_psnr": quality[-1]["heldout_psnr"] if quality else float("nan"),
        "final_psnr_min_view": quality[-1]["heldout_psnr_min"] if quality else float("nan"),
        "final_completeness": quality[-1]["completeness"] if quality else float("nan"),
        "final_accuracy": quality[-1]["accuracy"] if quality else float("nan"),
        "psnr_at_50": quality_at(quality, 50, "heldout_psnr"),
        "psnr_at_100": quality_at(quality, 100, "heldout_psnr"),
        "psnr_auc": float(np.nanmean(curve)) if curve.size else float("nan"),
        "phi_min_deg": float(np.nanmin(phi)),
        "phi_max_deg": float(np.nanmax(phi)),
        "rim_frac": float(np.mean(phi < RIM_DEG)),
        "high_frac": float(np.mean(phi > 80.0)),
        "lon_bins": int(last.get("longitude_bins_visited", 0)),
        "view_coverage": float(last.get("nbv_stop_view_coverage", np.nan)),
        "speed_median_mps": float(np.nanmedian([r.get("linear_speed_applied", np.nan) for r in moving]))
        if moving
        else 0.0,
        "limiter_frac": float(np.mean(lin_raw > vmax * (1 + 1e-9))) if log else float("nan"),
        "speed_max_mps": float(np.nanmax(lin)) if log else float("nan"),
        "mask_kept": float(np.nanmedian([r.get("nbv_mask_kept_ratio", np.nan) for r in log]))
        if log and "nbv_mask_kept_ratio" in log[0]
        else float("nan"),
        "feas_active_frac": float(np.mean([bool(r.get("nbv_feasibility_active")) for r in log]))
        if log and "nbv_feasibility_active" in log[0]
        else float("nan"),
        "_curve_steps": steps,
        "_curve_psnr": psnr,
        "_curve_compl": [r["completeness"] for r in quality],
        "_phi": phi.tolist(),
        "_theta": np.degrees([r["current_theta"] for r in log]).tolist() if log else [],
    }


def aggregate(values: list[float]) -> tuple[float, float]:
    arr = np.array([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=0))


def fmt(mean: float, std: float, digits: int = 2) -> str:
    if math.isnan(mean):
        return "–"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def main(root: Path) -> None:
    budget = 200
    runs: dict[str, dict[str, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for summary in sorted(root.glob("*/*/*/summary.json")):
        run_dir = summary.parent
        scene, arm, start = run_dir.parts[-3:]
        runs[scene][arm][start] = run_metrics(run_dir, budget)

    report = root / "report"
    report.mkdir(exist_ok=True)
    cols = [
        ("final_psnr", "终态PSNR(dB)", 2),
        ("psnr_auc", "PSNR曲线均值(dB)", 2),
        ("final_completeness", "完整度", 3),
        ("steps_run", "步数", 0),
        ("path_m", "路径(m)", 2),
        ("phi_min_deg", "最低仰角(°)", 1),
        ("rim_frac", "边缘占比(φ<15°)", 2),
        ("lon_bins", "经度分区/18", 1),
        ("view_coverage", "视角覆盖", 2),
        ("limiter_frac", "限幅占比", 2),
        ("speed_median_mps", "中位速度(m/s)", 3),
    ]
    lines = []
    for scene in sorted(runs):
        lines.append(f"\n### 场景：{scene}\n")
        lines.append("| 臂 | " + " | ".join(c[1] for c in cols) + " | 停止原因 |")
        lines.append("|---" * (len(cols) + 2) + "|")
        for arm in [a for a in ARM_ORDER if a in runs[scene]]:
            per_start = runs[scene][arm]
            cells = []
            for key, _, digits in cols:
                cells.append(fmt(*aggregate([m[key] for m in per_start.values()]), digits))
            reasons = ",".join(f"{s}:{m['stop_reason']}" for s, m in sorted(per_start.items()))
            lines.append(f"| {arm} | " + " | ".join(cells) + f" | {reasons} |")
        lines.append("\n逐起点（终态 PSNR / 最低仰角 / 步数）：\n")
        starts = sorted({s for arm in runs[scene].values() for s in arm})
        lines.append("| 臂 | " + " | ".join(starts) + " |")
        lines.append("|---" * (len(starts) + 1) + "|")
        for arm in [a for a in ARM_ORDER if a in runs[scene]]:
            cells = []
            for s in starts:
                m = runs[scene][arm].get(s)
                cells.append(
                    "–" if m is None else f"{m['final_psnr']:.2f} / {m['phi_min_deg']:.1f}° / {m['steps_run']}"
                )
            lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    (report / "tables.md").write_text("\n".join(lines) + "\n")
    clean = {
        scene: {arm: {s: {k: v for k, v in m.items() if not k.startswith("_")} for s, m in arms.items()}
                for arm, arms in by_arm.items()}
        for scene, by_arm in runs.items()
    }
    (report / "metrics.json").write_text(json.dumps(clean, indent=1, ensure_ascii=False))
    print("\n".join(lines))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for scene, by_arm in runs.items():
        starts = sorted({s for arm in by_arm.values() for s in arm})
        fig, axes = plt.subplots(2, len(starts), figsize=(5 * len(starts), 8), squeeze=False)
        for j, s in enumerate(starts):
            for arm in [a for a in ARM_ORDER if a in by_arm]:
                m = by_arm[arm].get(s)
                if m is None:
                    continue
                style = "-" if arm in ("legacy", "full", "tangent_only") else "--"
                lw = 2.2 if arm in ("legacy", "full") else 1.0
                axes[0, j].plot(m["_curve_steps"], m["_curve_psnr"], style, lw=lw, label=arm)
                axes[1, j].plot(m["_phi"], style, lw=lw, label=arm)
            axes[0, j].set_title(f"{scene} / {s}: held-out PSNR")
            axes[0, j].set_xlabel("step")
            axes[0, j].set_ylabel("dB")
            axes[1, j].axhline(RIM_DEG, color="k", lw=0.6, ls=":")
            axes[1, j].set_title(f"{scene} / {s}: elevation φ")
            axes[1, j].set_xlabel("step")
            axes[1, j].set_ylabel("deg")
        axes[0, 0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(report / f"{scene}_curves.png", dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "sim/sim_outputs/nbv_ablation"))
