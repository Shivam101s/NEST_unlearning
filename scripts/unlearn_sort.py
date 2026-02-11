#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Post-hoc ranking of unlearning iterations by a unified score (UES).

- Scans per-celeb run folders for: log_*NEURON-GLOBAL.txt
- Parses baseline + [GLOBAL] iteration lines (your exact format)
- Computes UES per iteration:
    UES = alpha * mean_rel_drop(forget@1, forget@5)
          - (1 - alpha) * mean_rel_drop(test@1, test@5, celeb@1, celeb@5)
- Optionally enforces floors on retain metrics (absolute 0–1 tolerance)
- Writes: sorted_by_ues.txt (descending UES) and best_by_ues.txt per run
"""

import re, json
from pathlib import Path
from typing import Dict, List, Optional

# --------------------
# CONFIG: edit here
# --------------------
BASE_DIR = Path("/home/rania/SLUG/results/neuron importance")
CELEBS   = ["Elon_Musk", "Taylor_Swift", "Jeff_Bezos", "Mark_Zuckerberg", "Kim_Kardashian"]
RUN_SUBDIR = "NEURON_Si_GLOBAL"     # where your log_*NEURON-GLOBAL.txt lives per celeb

ALPHA = 0.5         # weight on forgetting (0..1)
TOL_ABS = 0.01       # absolute floor tolerance (0..1). 0.01 = 1 percentage point
REQUIRE_FLOORS = False
PREFER_LOWER_FORGET = True  # tie-break: prefer lower (fgt5,fgt1) when UES ties

# --------------------
# Parsers
# --------------------
P_BASE = re.compile(
    r"baseline:\s*fgt@1=(?P<f1>[-+.\deE]+)\s*fgt@5=(?P<f5>[-+.\deE]+)\s*\|\s*"
    r"celeb@1=(?P<c1>[-+.\deE]+)\s*celeb@5=(?P<c5>[-+.\deE]+)\s*\|\s*"
    r"test@1=(?P<t1>[-+.\deE]+)\s*test@5=(?P<t5>[-+.\deE]+)"
)

P_IT = re.compile(
    r"\[GLOBAL\]\s*iter:(?P<i>\d+)\s*step:(?P<s>[-+.\deE]+)\s*\|\s*"
    r"fgt@1:(?P<f1>[-+.\deE]+)\s*fgt@5:(?P<f5>[-+.\deE]+)\s*\|\s*"
    r"celeb@1:(?P<c1>[-+.\deE]+)\s*celeb@5:(?P<c5>[-+.\deE]+)\s*\|\s*"
    r"test@1:(?P<t1>[-+.\deE]+)\s*test@5:(?P<t5>[-+.\deE]+)"
)

def parse_log(log_path: Path):
    txt = log_path.read_text()
    m = P_BASE.search(txt)
    if not m:
        raise RuntimeError(f"Could not parse baseline in {log_path}")
    base = {
        "fgt1": float(m.group("f1")), "fgt5": float(m.group("f5")),
        "celeb1": float(m.group("c1")), "celeb5": float(m.group("c5")),
        "test1": float(m.group("t1")), "test5": float(m.group("t5")),
    }
    iters: List[Dict] = []
    for mm in P_IT.finditer(txt):
        iters.append({
            "iter": int(mm.group("i")),
            "step": float(mm.group("s")),
            "fgt1": float(mm.group("f1")), "fgt5": float(mm.group("f5")),
            "celeb1": float(mm.group("c1")), "celeb5": float(mm.group("c5")),
            "test1": float(mm.group("t1")), "test5": float(mm.group("t5")),
        })
    if not iters:
        raise RuntimeError(f"No iteration lines parsed in {log_path}")
    return base, iters

# --------------------
# Scoring
# --------------------
def rel_drop(curr: float, base: float, eps: float = 1e-6) -> float:
    """Relative drop in [0,1]; 0 if improved or equal."""
    if base <= eps:
        return 0.0 if curr <= base else min(1.0, (curr - base) / max(eps, curr))
    d = max(0.0, base - curr) / max(eps, base)
    return min(1.0, d)

def mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0

def compute_ues(base: Dict[str,float], it: Dict[str,float], alpha: float = ALPHA) -> float:
    forget_gain = mean([rel_drop(it["fgt1"], base["fgt1"]), rel_drop(it["fgt5"], base["fgt5"])])
    retain_loss = mean([
        rel_drop(it["test1"],  base["test1"]),
        rel_drop(it["test5"],  base["test5"]),
        rel_drop(it["celeb1"], base["celeb1"]),
        rel_drop(it["celeb5"], base["celeb5"]),
    ])
    return alpha * forget_gain - (1.0 - alpha) * retain_loss

def floors_ok(base: Dict[str,float], it: Dict[str,float], tol_abs: float = TOL_ABS) -> bool:
    return (
        it["test1"]  >= max(base["test1"]  - tol_abs, 0.0) and
        it["test5"]  >= max(base["test5"]  - tol_abs, 0.0) and
        it["celeb1"] >= max(base["celeb1"] - tol_abs, 0.0) and
        it["celeb5"] >= max(base["celeb5"] - tol_abs, 0.0)
    )

# --------------------
# Main orchestration
# --------------------
def rank_one_run(run_dir: Path):
    logs = sorted(run_dir.glob("log_*NEURON-GLOBAL.txt"))
    if not logs:
        print(f"[WARN] No logs in {run_dir}")
        return
    logp = logs[0]  # if multiple, pick first; adjust if needed
    base, iters = parse_log(logp)

    rows = []
    for it in iters:
        if REQUIRE_FLOORS and not floors_ok(base, it, TOL_ABS):
            continue
        ues = compute_ues(base, it, ALPHA)
        # tie-breaker: prefer lower forget if flagged
        tie_break = - (it["fgt5"] * 1000.0 + it["fgt1"]) if PREFER_LOWER_FORGET else 0.0
        rows.append((ues, tie_break, it))

    if not rows:
        print(f"[WARN] No feasible iterations after floors in {run_dir}. "
              f"Consider lowering TOL_ABS or set REQUIRE_FLOORS=False.")
        return

    rows.sort(key=lambda x: (x[0], x[1]), reverse=True)  # highest UES first; tie-break

    # Write sorted file
    out_sorted = run_dir / "sorted_by_ues.txt"
    with out_sorted.open("w") as f:
        f.write(f"# source_log: {logp}\n")
        f.write(f"# alpha={ALPHA} tol_abs={TOL_ABS} require_floors={REQUIRE_FLOORS} prefer_lower_forget={PREFER_LOWER_FORGET}\n")
        f.write(f"# baseline: fgt@1={base['fgt1']:.4f} fgt@5={base['fgt5']:.4f} | "
                f"celeb@1={base['celeb1']:.4f} celeb@5={base['celeb5']:.4f} | "
                f"test@1={base['test1']:.4f} test@5={base['test5']:.4f}\n")
        f.write("# UES  iter  step        fgt@1  fgt@5  celeb@1  celeb@5  test@1  test@5\n")
        for ues, tb, it in rows:
            f.write(f"{ues: .6f}  {it['iter']:>4}  {it['step']:>+10.6f}  "
                    f"{it['fgt1']:.4f}  {it['fgt5']:.4f}  "
                    f"{it['celeb1']:.4f}  {it['celeb5']:.4f}  "
                    f"{it['test1']:.4f}  {it['test5']:.4f}\n")

    # Write a one-liner best file
    best_ues, _, best_it = rows[0]
    out_best = run_dir / "best_by_ues.txt"
    out_best.write_text(
        f"best iter={best_it['iter']} step={best_it['step']:+.6f} UES={best_ues:.6f} | "
        f"fgt@1={best_it['fgt1']:.4f} fgt@5={best_it['fgt5']:.4f} | "
        f"celeb@1={best_it['celeb1']:.4f} celeb@5={best_it['celeb5']:.4f} | "
        f"test@1={best_it['test1']:.4f} test@5={best_it['test5']:.4f}\n"
    )

    print(f"[OK] Wrote {out_sorted}")
    print(f"[OK] Wrote {out_best}")

def main():
    for celeb in CELEBS:
        run_dir = BASE_DIR / celeb / RUN_SUBDIR
        if not run_dir.exists():
            print(f"[WARN] Missing run dir: {run_dir}")
            continue
        rank_one_run(run_dir)

if __name__ == "__main__":
    main()
