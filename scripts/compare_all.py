# scripts/compare_all_v2.py
import re
from pathlib import Path

RESULT_ROOT = Path("/home/rania/SLUG/results/slug_prompt_metrics")

# =========================
# Common parsers (robust)
# =========================
NUMBER = r"[eE\+\-0-9\.]+"
FLOAT  = r"[0-9]*\.?[0-9]+"

# Optional "nbr@1, nbr@5" block (with trailing comma). Groups 1 & 2 when present.
NBR_BLOCK = (
    r"(?:nbr@1:\s*(" + FLOAT + r")\s*,\s*nbr@5:\s*(" + FLOAT + r")\s*,\s*)?"
)

# Allow any number of "Key: value," (letters/underscores only) before MIA
# (we keep it in case you add new extras; nbr uses @, so we capture it separately above)
EXTRAS = r"(?:[A-Za-z_]+:\s*" + NUMBER + r",\s*)*"

# MIA may appear as "MIA:" or "MIA(%):"
MIA_FIELD = r"MIA(?:\(%\))?:\s*(" + FLOAT + r")\s*±\s*(" + FLOAT + r")"

# Baseline lines (no ratio)
BASELINE_LINE = re.compile(
    r"iter:\s*(\d+),\s*"
    r"(?:ratio:\s*" + NUMBER + r",\s*)?"  # tolerate ratio if present
    r"fgt_acc@1:\s*(" + FLOAT + r"),\s*fgt_acc@5:\s*(" + FLOAT + r"),\s*"
    r"celeba100@1:\s*(" + FLOAT + r"),\s*celeba100@5:\s*(" + FLOAT + r"),\s*"
    r"test_acc@1:\s*(" + FLOAT + r"),\s*test_acc@5:\s*(" + FLOAT + r"),\s*"
    + NBR_BLOCK + EXTRAS + MIA_FIELD
)

# SLUG lines (have ratio)
SLUG_LINE = re.compile(
    r"iter:\s*(\d+),\s*ratio:\s*(" + NUMBER + r"),\s*"
    r"fgt_acc@1:\s*(" + FLOAT + r"),\s*fgt_acc@5:\s*(" + FLOAT + r"),\s*"
    r"celeba100@1:\s*(" + FLOAT + r"),\s*celeba100@5:\s*(" + FLOAT + r"),\s*"
    r"test_acc@1:\s*(" + FLOAT + r"),\s*test_acc@5:\s*(" + FLOAT + r"),\s*"
    + NBR_BLOCK + EXTRAS + MIA_FIELD
)

def _row_from_match(m, has_ratio=False):
    """
    Build a row dict from a regex match (both baseline and slug).
    Group indices depend on whether ratio is captured.
    """
    # Base groups (before nbr)
    g = m.groups()
    idx = 0
    row = {}

    row["iter"] = int(g[idx]); idx += 1

    if has_ratio:
        row["ratio"] = float(g[idx]); idx += 1

    row["fgt_acc1"]   = float(g[idx]); idx += 1
    row["fgt_acc5"]   = float(g[idx]); idx += 1
    row["celeb_top1"] = float(g[idx]); idx += 1
    row["celeb_top5"] = float(g[idx]); idx += 1
    row["test_acc1"]  = float(g[idx]); idx += 1
    row["test_acc5"]  = float(g[idx]); idx += 1

    # Optional nbr block (2 groups or None)
    nbr1_s = g[idx]; idx += 1
    nbr5_s = g[idx]; idx += 1
    row["nbr_top1"] = float(nbr1_s) if nbr1_s is not None else None
    row["nbr_top5"] = float(nbr5_s) if nbr5_s is not None else None

    # Extras (if any) are eaten by EXTRAS and not captured — that’s fine.

    # MIA (2 groups)
    row["MIA_mean"] = float(g[idx]); idx += 1
    row["MIA_std"]  = float(g[idx]); idx += 1

    return row

def parse_baseline_file(file_path: Path):
    rows = []
    with open(file_path, "r", errors="ignore") as f:
        for line in f:
            m = BASELINE_LINE.search(line)
            if not m:
                continue
            try:
                rows.append(_row_from_match(m, has_ratio=False))
            except Exception:
                # Skip malformed lines gracefully
                continue
    return rows

def parse_slug_log(file_path: Path):
    rows = []
    with open(file_path, "r", errors="ignore") as f:
        for line in f:
            m = SLUG_LINE.search(line)
            if not m:
                continue
            try:
                rows.append(_row_from_match(m, has_ratio=True))
            except Exception:
                continue
    return rows

# =========================
# Picking rules (paper-aligned)
# =========================
def pick_from_baseline_rows(rows):
    """For a baseline log, pick first row with fgt_acc1==0; else minimal fgt_acc1."""
    if not rows:
        return None
    first_zero = next((r for r in rows if r["fgt_acc1"] == 0.0), None)
    if first_zero is not None:
        return first_zero
    return min(rows, key=lambda r: r["fgt_acc1"])

def pick_best_slug_row(rows, eps=1e-9):
    if not rows:
        return None

    # 1) Minimize forgetting (prefer @5 first, then @1)
    fa5_min = min(r["fgt_acc5"] for r in rows)
    cand = [r for r in rows if r["fgt_acc5"] <= fa5_min + eps]

    fa1_min = min(r["fgt_acc1"] for r in cand)
    cand = [r for r in cand if r["fgt_acc1"] <= fa1_min + eps]

    # 2) Maximize utility (average of IN@1 and CelebA@1)
    def utility(r): return 0.5 * (r["test_acc1"] + r["celeb_top1"])
    util_max = max(utility(r) for r in cand)
    cand = [r for r in cand if abs(utility(r) - util_max) <= eps]

    # 3) Tie-breakers: lower MIA_mean, then smaller |ratio|, then earlier iter
    return min(cand, key=lambda r: (r["MIA_mean"], abs(r.get("ratio", 0.0)), r["iter"]))

# =========================
# Utilities
# =========================
def mean_of_dicts(dicts, keys_scalar, keys_pair=None):
    """Average selected keys over a list of dicts (skip None)."""
    if not dicts:
        return {}
    n = len(dicts)
    out = {}
    for k in keys_scalar:
        vals = [d[k] for d in dicts if d.get(k) is not None]
        out[k] = (sum(vals) / len(vals)) if vals else None
    if keys_pair:
        for k in keys_pair:
            vals = [d[k] for d in dicts if d.get(k) is not None]
            out[k] = (sum(vals) / len(vals)) if vals else None
    return out

def fmt_pct(x):
    return "n/a" if x is None else f"{x*100:.2f}%"

def fmt_num(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"

# =========================
# Main comparison
# =========================
def main():
    # -------- Discover the celeb set from SLUG folders --------
    slug_dirs = sorted(RESULT_ROOT.glob("slug_*"))
    celebs = [d.name.replace("slug_", "") for d in slug_dirs]
    if not celebs:
        print("No SLUG folders found (/home/rania/SLUG/results/slug_prompt_metrics/slug_*).")
        return

    # -------- Baselines averaged across the same celebs --------
    methods = ["raw", "ft", "ga", "gaft", "salun", "ssd"]
    for method in methods:
        per_celeb_picks = []
        for celeb in celebs:
            files = sorted(RESULT_ROOT.glob(f"*_{celeb}_*_{method}*.txt"))
            best_for_celeb = None
            for f in files:
                rows = parse_baseline_file(f)
                pick = pick_from_baseline_rows(rows)
                if pick:
                    if (best_for_celeb is None or
                        pick["fgt_acc1"] < best_for_celeb["fgt_acc1"] or
                        (pick["fgt_acc1"] == best_for_celeb["fgt_acc1"] and pick["test_acc1"] > best_for_celeb["test_acc1"])):
                        best_for_celeb = pick
            if best_for_celeb:
                per_celeb_picks.append(best_for_celeb)

        if not per_celeb_picks:
            print(f"{method}: ")
        else:
            avg = mean_of_dicts(
                per_celeb_picks,
                keys_scalar=[
                    "iter","fgt_acc1","fgt_acc5",
                    "test_acc1","test_acc5",
                    "celeb_top1","celeb_top5",
                    "nbr_top1","nbr_top5",   # will be None if not logged
                ],
                keys_pair=["MIA_mean","MIA_std"]
            )
            print(
                f"{method}: "
                f"iterations: {fmt_num(avg['iter'],0)}, "
                f"fgt_acc1: {fmt_pct(avg['fgt_acc1'])}, "
                f"fgt_acc5: {fmt_pct(avg['fgt_acc5'])}, "
                f"test_acc1: {fmt_pct(avg['test_acc1'])}, "
                f"test_acc5: {fmt_pct(avg['test_acc5'])}, "
                f"MIA: {fmt_num(avg['MIA_mean'])} ± {fmt_num(avg['MIA_std'])} "
                f"celeb_top1: {fmt_pct(avg['celeb_top1'])}, "
                f"celeb_top5: {fmt_pct(avg['celeb_top5'])}, "
                f"nbr_top1: {fmt_pct(avg['nbr_top1'])}, "
                f"nbr_top5: {fmt_pct(avg['nbr_top5'])}"
            )

    # -------- SLUG: per-celeb winner, then average; also global winner --------
    per_celeb_slug_winners = []
    global_layer_rows = []  # (file_name, celeb, best_row)

    for d in slug_dirs:
        celeb = d.name.replace("slug_", "")
        layer_bests = []
        for logf in d.glob("log_*.txt"):
            rows = parse_slug_log(logf)
            best = pick_best_slug_row(rows)
            if best:
                layer_bests.append((logf.name, best))
                global_layer_rows.append((logf.name, celeb, best))

        if not layer_bests:
            print(f"[{celeb}] No valid rows found.")
            continue

        name_c, best_c = max(
            layer_bests,
            key=lambda t: (
                (t[1]["fgt_acc5"] == 0.0, t[1]["fgt_acc1"] == 0.0),
                0.5*(t[1]["test_acc1"] + t[1]["celeb_top1"]),
                -t[1]["MIA_mean"]
            )
        )
        per_celeb_slug_winners.append({
            "celeb": celeb,
            "file": name_c,
            **best_c
        })

    # Print per-celeb SLUG winners
    if per_celeb_slug_winners:
        print("\n--- SLUG per-celeb winners ---")
        for w in per_celeb_slug_winners:
            print(
                f"{w['celeb']}: iter={w['iter']}, "
                f"fgt@1={fmt_pct(w['fgt_acc1'])}, fgt@5={fmt_pct(w['fgt_acc5'])}, "
                f"IN@1={fmt_pct(w['test_acc1'])}, CA@1={fmt_pct(w['celeb_top1'])}, "
                f"nbr@1={fmt_pct(w.get('nbr_top1'))}, nbr@5={fmt_pct(w.get('nbr_top5'))}, "
                f"MIA={fmt_num(w['MIA_mean'])}±{fmt_num(w['MIA_std'])} [{w['file']}]"
            )

        # Average SLUG across celebs
        avg_slug = mean_of_dicts(
            per_celeb_slug_winners,
            keys_scalar=[
                "iter","fgt_acc1","fgt_acc5",
                "test_acc1","test_acc5",
                "celeb_top1","celeb_top5",
                "nbr_top1","nbr_top5",
            ],
            keys_pair=["MIA_mean","MIA_std"]
        )
        print(
            "\nslug_avg: "
            f"iterations: {fmt_num(avg_slug['iter'],0)}, "
            f"fgt_acc1: {fmt_pct(avg_slug['fgt_acc1'])}, "
            f"fgt_acc5: {fmt_pct(avg_slug['fgt_acc5'])}, "
            f"test_acc1: {fmt_pct(avg_slug['test_acc1'])}, "
            f"test_acc5: {fmt_pct(avg_slug['test_acc5'])}, "
            f"MIA: {fmt_num(avg_slug['MIA_mean'])} ± {fmt_num(avg_slug['MIA_std'])} "
            f"celeb_top1: {fmt_pct(avg_slug['celeb_top1'])}, "
            f"celeb_top5: {fmt_pct(avg_slug['celeb_top5'])}, "
            f"nbr_top1: {fmt_pct(avg_slug['nbr_top1'])}, "
            f"nbr_top5: {fmt_pct(avg_slug['nbr_top5'])}"
        )
    else:
        print("slug_avg: ")

    # Single global SLUG winner across ALL celebs/layers (optional)
    if global_layer_rows:
        g_name, g_celeb, g_best = max(
            global_layer_rows,
            key=lambda t: (
                (t[2]["fgt_acc5"] == 0.0, t[2]["fgt_acc1"] == 0.0),
                0.5*(t[2]["test_acc1"] + t[2]["celeb_top1"]),
                -t[2]["MIA_mean"]
            )
        )
        print(
            "\nslug_global_one_winner: "
            f"iterations: {int(g_best['iter'])}, "
            f"fgt_acc1: {fmt_pct(g_best['fgt_acc1'])}, "
            f"fgt_acc5: {fmt_pct(g_best['fgt_acc5'])}, "
            f"test_acc1: {fmt_pct(g_best['test_acc1'])}, "
            f"test_acc5: {fmt_pct(g_best['test_acc5'])}, "
            f"MIA: {fmt_num(g_best['MIA_mean'])} ± {fmt_num(g_best['MIA_std'])} "
            f"celeb_top1: {fmt_pct(g_best['celeb_top1'])}, "
            f"celeb_top5: {fmt_pct(g_best['celeb_top5'])}, "
            f"nbr_top1: {fmt_pct(g_best.get('nbr_top1'))}, "
            f"nbr_top5: {fmt_pct(g_best.get('nbr_top5'))} "
            f"[{g_celeb}/{g_name}]"
        )
    else:
        print("slug_global_one_winner: ")

if __name__ == "__main__":
    main()
