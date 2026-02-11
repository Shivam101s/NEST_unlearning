# scripts/compare_all.py
import re
from pathlib import Path

RESULT_ROOT = Path("results")

# =========================
# Baseline (raw/ft/ga/gaft/salun/ssd)
# =========================
BASELINE_LINE = re.compile(
    r'iter:\s*(\d+),\s*'
    r'fgt_acc@1:\s*([\d\.]+),\s*fgt_acc@5:\s*([\d\.]+),\s*'
    r'celeba100@1:\s*([\d\.]+),\s*celeba100@5:\s*([\d\.]+),\s*'
    r'test_acc@1:\s*([\d\.]+),\s*test_acc@5:\s*([\d\.]+),\s*'
    r'MIA:\s*([\d\.]+)±([\d\.]+)'
)

def parse_baseline_file(file_path: Path):
    rows = []
    with open(file_path, "r", errors="ignore") as f:
        for line in f:
            m = BASELINE_LINE.search(line)
            if not m:
                continue
            rows.append({
                "iter": int(m.group(1)),
                "fgt_acc1": float(m.group(2)),
                "fgt_acc5": float(m.group(3)),
                "celeb_top1": float(m.group(4)),
                "celeb_top5": float(m.group(5)),
                "test_acc1": float(m.group(6)),
                "test_acc5": float(m.group(7)),
                "MIA_mean": float(m.group(8)),
                "MIA_std": float(m.group(9)),
            })
    return rows

def pick_from_baseline_rows(rows):
    """
    For each baseline log, pick the row that first achieves fgt_acc1==0;
    otherwise pick the row with minimal fgt_acc1.
    """
    if not rows:
        return None
    first_zero = next((r for r in rows if r["fgt_acc1"] == 0.0), None)
    if first_zero is not None:
        return first_zero
    return min(rows, key=lambda r: r["fgt_acc1"])

def summarize_baselines(result_root: Path):
    methods = ["raw", "ft", "ga", "gaft", "salun", "ssd"]
    for method in methods:
        # match *_method.txt and *_method_*txt (handles filenames with lr suffixes)
        files = sorted(result_root.glob(f"*_{method}*.txt"))
        picks = []
        for f in files:
            rows = parse_baseline_file(f)
            best = pick_from_baseline_rows(rows)
            if best:
                picks.append(best)
        if not picks:
            print(f"{method}: ")
            continue

        # simple mean across the 5 identities
        n = len(picks)
        mean = lambda key: sum(p[key] for p in picks) / n
        # MIA is two numbers
        mia_mean = sum(p["MIA_mean"] for p in picks) / n
        mia_std  = sum(p["MIA_std"]  for p in picks) / n

        print(
            f"{method}: "
            f"iterations: {mean('iter'):.0f}, "
            f"fgt_acc1: {mean('fgt_acc1')*100:.2f}%, "
            f"fgt_acc5: {mean('fgt_acc5')*100:.2f}%, "
            f"test_acc1: {mean('test_acc1')*100:.2f}%, "
            f"test_acc5: {mean('test_acc5')*100:.2f}%, "
            f"MIA: {mia_mean:.2f} ± {mia_std:.2f} "
            f"celeb_top1: {mean('celeb_top1')*100:.2f}%, "
            f"celeb_top5: {mean('celeb_top5')*100:.2f}%, "
        )

# =========================
# SLUG (per-layer logs, pick one winner)
# =========================
SLUG_LINE = re.compile(
    r'iter:\s*(\d+),\s*ratio:\s*([eE\+\-0-9\.]+),\s*'
    r'fgt_acc@1:\s*([\d\.]+),\s*fgt_acc@5:\s*([\d\.]+),\s*'
    r'celeba100@1:\s*([\d\.]+),\s*celeba100@5:\s*([\d\.]+),\s*'
    r'test_acc@1:\s*([\d\.]+),\s*test_acc@5:\s*([\d\.]+),\s*'
    r'MIA:\s*([\d\.]+)±([\d\.]+)'
)

def parse_slug_log(file_path: Path):
    rows = []
    with open(file_path, "r", errors="ignore") as f:
        for line in f:
            m = SLUG_LINE.search(line)
            if not m:
                continue
            rows.append({
                "iter": int(m.group(1)),
                "ratio": float(m.group(2)),
                "fgt_acc1": float(m.group(3)),
                "fgt_acc5": float(m.group(4)),
                "celeb_top1": float(m.group(5)),
                "celeb_top5": float(m.group(6)),
                "test_acc1": float(m.group(7)),
                "test_acc5": float(m.group(8)),
                "MIA_mean": float(m.group(9)),
                "MIA_std": float(m.group(10)),
            })
    return rows

def pick_best_slug_row(rows):
    """
    Paper-aligned selection for a single layer:
      1) Forgetting: prefer FA@5==0; inside that, prefer FA@1==0.
      2) Utility: maximize average of (ImageNet@1, CelebA@1).
      3) Privacy: minimize MIA_mean.
    """
    if not rows:
        return None

    def forget_key(r):
        # higher is better
        return (r["fgt_acc5"] == 0.0, r["fgt_acc1"] == 0.0)

    def utility_score(r):
        return 0.5 * (r["test_acc1"] + r["celeb_top1"])

    return max(rows, key=lambda r: (forget_key(r), utility_score(r), -r["MIA_mean"]))

def summarize_slug(result_root: Path):
    slug_dirs = sorted(result_root.glob("slug_*"))
    if not slug_dirs:
        print("slug: ")
        return

    per_layer_bests = []
    for d in slug_dirs:
        for logf in d.glob("log_*.txt"):
            rows = parse_slug_log(logf)
            best = pick_best_slug_row(rows)
            if best:
                per_layer_bests.append((logf.name, best))

    if not per_layer_bests:
        print("slug: ")
        return

    # Global SLUG winner across ALL layers considered
    winner_name, winner = max(
        per_layer_bests,
        key=lambda t: (
            (t[1]["fgt_acc5"] == 0.0, t[1]["fgt_acc1"] == 0.0),
            0.5 * (t[1]["test_acc1"] + t[1]["celeb_top1"]),
            -t[1]["MIA_mean"]
        )
    )

    print(
        "slug: "
        f"iterations: {winner['iter']:.0f}, "
        f"fgt_acc1: {winner['fgt_acc1']*100:.2f}%, "
        f"fgt_acc5: {winner['fgt_acc5']*100:.2f}%, "
        f"test_acc1: {winner['test_acc1']*100:.2f}%, "
        f"test_acc5: {winner['test_acc5']*100:.2f}%, "
        f"MIA: {winner['MIA_mean']:.2f} ± {winner['MIA_std']:.2f} "
        f"celeb_top1: {winner['celeb_top1']*100:.2f}%, "
        f"celeb_top5: {winner['celeb_top5']*100:.2f}%, "
        f"[{winner_name}]"
    )

# =========================
# Main
# =========================
def main():
    summarize_baselines(RESULT_ROOT)
    summarize_slug(RESULT_ROOT)

if __name__ == "__main__":
    main()
