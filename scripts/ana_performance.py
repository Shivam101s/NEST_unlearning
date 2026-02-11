#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize unlearning results (per method) with averaged UES (using GLOBAL baselines when available).

- For each method, scan logs (default: results/*.txt; use --recurse to rglob).
- For each file, select the row where fgt_acc@1 == 0 (first occurrence),
  otherwise the row with the minimum fgt_acc@1.
- Baseline for UES:
    * If a GLOBAL baseline for the file's concept exists -> use that
    * else fallback to per-file baseline (iter==0 or earliest)
- Compute UES (your definition) for that file's selected row vs chosen baseline.
- Average selected rows across files (per method), including UES.
- Print metrics as percentages (except MIA mean±std and UES scalar).

CLI:
  --root PATH            root folder (default: results)
  --methods LIST         comma-separated (default: raw,ft,ga,gaft,salun,ssd)
  --recurse              search subfolders recursively
  --alpha FLOAT          UES alpha (default 0.5)
  --baseline_json PATH   optional JSON: {"Concept": {"f1":..,"f5":..,"c1":..,"c5":..,"t1":..,"t5":..}, ...}
  --concept_keys LIST    comma-separated list of concept keys to look for in path (defaults to keys of built-in map)
  --debug                print files found and parsing counts
"""

import re
import math
import json
import argparse
from pathlib import Path

# -------------------------- DEFAULT GLOBAL BASELINES --------------------------
# These are the "raw/original" baselines you provided.
DEFAULT_GLOBAL_BASELINES = {
    "Elon_Musk":        {"f1": 0.6667, "f5": 0.9444, "c1": 0.6166, "c5": 0.8248, "t1": 0.6008, "t5": 0.8548},
    "Mark_Zuckerberg":  {"f1": 0.9500, "f5": 0.9500, "c1": 0.6166, "c5": 0.8248, "t1": 0.6008, "t5": 0.8548},
    "Jeff_Bezos":       {"f1": 0.6000, "f5": 1.0000, "c1": 0.6166, "c5": 0.8248, "t1": 0.6008, "t5": 0.8548},
    "Kim_Kardashian":   {"f1": 0.5714, "f5": 0.7619, "c1": 0.6166, "c5": 0.8248, "t1": 0.6008, "t5": 0.8548},
}

# -------------------------- UES helpers --------------------------

def _rel_drop(current: float, base: float) -> float:
    if not isinstance(base, (int, float)) or not math.isfinite(base) or abs(base) < 1e-12:
        return 0.0
    if not isinstance(current, (int, float)) or not math.isfinite(current):
        return 0.0
    return (base - current) / base

def _ues_score(
    f1, f5, t1, t5, c1, c5,
    base_f1, base_f5, base_t1, base_t5, base_c1, base_c5,
    alpha: float
) -> float:
    # Shivam's UES definition
    forget_gain = 0.5 * (_rel_drop(f1, base_f1) + _rel_drop(f5, base_f5))
    retain_loss = 0.25 * (
        _rel_drop(t1, base_t1) + _rel_drop(t5, base_t5) +
        _rel_drop(c1, base_c1) + _rel_drop(c5, base_c5)
    )
    return alpha * forget_gain - (1.0 - alpha) * retain_loss

# -------------------------- averaging --------------------------

def calculate_averages(rows):
    """Average a list of dicts (drop keys that are None). Supports tuples (MIA mean±std)."""
    sums, counts = {}, {}
    tuple_sums, tuple_counts = {}, {}

    for row in rows:
        for k, v in row.items():
            if v is None:
                continue
            if isinstance(v, (int, float)):
                sums[k] = sums.get(k, 0.0) + float(v)
                counts[k] = counts.get(k, 0) + 1
            elif isinstance(v, tuple) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
                a, b = v
                ta, tb = tuple_sums.get(k, (0.0, 0.0))
                tuple_sums[k] = (ta + float(a), tb + float(b))
                tuple_counts[k] = tuple_counts.get(k, 0) + 1

    avgs = {}
    for k, s in sums.items():
        avgs[k] = s / max(1, counts.get(k, 1))
    for k, (sa, sb) in tuple_sums.items():
        c = max(1, tuple_counts.get(k, 1))
        avgs[k] = (sa / c, sb / c)
    return avgs

# -------------------------- parsing --------------------------

PAT_LINE = re.compile(
    r'iter:\s*(?P<iter>\d+)\s*,\s*'
    r'(?:(?:ratio|lambda):\s*[-+eE0-9\.]+\s*,\s*)?'  # OPTIONAL ratio/lambda
    r'fgt_acc@1:\s*(?P<f1>[-+eE0-9\.]+)\s*,\s*'
    r'fgt_acc@5:\s*(?P<f5>[-+eE0-9\.]+)\s*,\s*'
    r'celeba100@1:\s*(?P<c1>[-+eE0-9\.]+)\s*,\s*'
    r'celeba100@5:\s*(?P<c5>[-+eE0-9\.]+)\s*,\s*'
    r'test_acc@1:\s*(?P<t1>[-+eE0-9\.]+)\s*,\s*'
    r'test_acc@5:\s*(?P<t5>[-+eE0-9\.]+)'
    r'(?:\s*,\s*nbr@1:\s*(?P<n1>[-+eE0-9\.]+))?'
    r'(?:\s*,\s*nbr@5:\s*(?P<n5>[-+eE0-9\.]+))?'
    r'(?:\s*,\s*TISI:\s*(?P<TISI>[-+eE0-9\.]+))?'
    r'(?:\s*,\s*PAR:\s*(?P<PAR>[-+eE0-9\.]+))?'
    r'(?:\s*,\s*EGR:\s*(?P<EGR>[-+eE0-9\.]+))?'
    r'\s*,\s*MIA(?:\(%\))?:\s*(?P<mia_m>[-+eE0-9\.]+)\s*±\s*(?P<mia_s>[-+eE0-9\.]+)',
    re.IGNORECASE
)

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def parse_file(file_path: Path):
    """Parse one file into column lists + iter list."""
    iters = []
    fgt_acc1, fgt_acc5 = [], []
    celeb_top1, celeb_top5 = [], []
    test_acc1, test_acc5 = [], []
    nbr_top1, nbr_top5 = [], []
    tisi, par, egr = [], [], []
    mia_list = []

    with open(file_path, 'r', errors='ignore') as fh:
        for line in fh:
            m = PAT_LINE.search(line)
            if not m:
                continue
            g = m.groupdict()
            iters.append(int(g['iter']))
            fgt_acc1.append(_to_float(g['f1'])); fgt_acc5.append(_to_float(g['f5']))
            celeb_top1.append(_to_float(g['c1'])); celeb_top5.append(_to_float(g['c5']))
            test_acc1.append(_to_float(g['t1'])); test_acc5.append(_to_float(g['t5']))
            nbr_top1.append(_to_float(g.get('n1')) if g.get('n1') else None)
            nbr_top5.append(_to_float(g.get('n5')) if g.get('n5') else None)
            tisi.append(_to_float(g.get('TISI')) if g.get('TISI') else None)
            par.append(_to_float(g.get('PAR')) if g.get('PAR') else None)
            egr.append(_to_float(g.get('EGR')) if g.get('EGR') else None)
            mia_list.append((_to_float(g['mia_m']), _to_float(g['mia_s'])))

    return {
        'iter': iters,
        'fgt_acc@1': fgt_acc1,
        'fgt_acc@5': fgt_acc5,
        'celeba100@1': celeb_top1,
        'celeba100@5': celeb_top5,
        'test_acc@1': test_acc1,
        'test_acc@5': test_acc5,
        'nbr@1': nbr_top1,
        'nbr@5': nbr_top5,
        'TISI': tisi,
        'PAR': par,
        'EGR': egr,
        'MIA': mia_list,
    }

# -------------------------- selection + baselines --------------------------

def _baseline_index(results):
    iters = results.get('iter', [])
    if not iters:
        return 0
    for i, it in enumerate(iters):
        if it == 0:
            return i
    # else earliest
    min_i = min(range(len(iters)), key=lambda k: iters[k])
    return min_i

def _row_at(results, idx: int):
    row = {}
    for k, vlist in results.items():
        if k == 'iter':
            continue
        if not vlist or idx >= len(vlist):
            row[k] = None
        else:
            row[k] = vlist[idx]
    return row

def select_row(results):
    """Return index of selected row using: first fgt@1==0 else argmin fgt@1."""
    fa1 = results.get('fgt_acc@1', [])
    if not fa1:
        return None
    sel_idx = next((i for i, v in enumerate(fa1) if v == 0.0), None)
    if sel_idx is None:
        valid = [(i, v) for i, v in enumerate(fa1) if isinstance(v, (int, float))]
        if not valid:
            return None
        sel_idx = min(valid, key=lambda x: x[1])[0]
    return sel_idx

def infer_concept_from_path(p: Path, concept_keys: list[str]) -> str | None:
    low = str(p).lower()
    # Match by simple substring of a normalized key (underscores/hyphens/space-insensitive)
    for key in concept_keys:
        k_low = key.lower().replace(" ", "_")
        if k_low in low:
            return key
    return None

# -------------------------- main --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results"))
    ap.add_argument("--methods", type=str, default="raw,ft,ga,gaft,salun,ssd",
                    help="Comma-separated method tokens to match in filenames")
    ap.add_argument("--recurse", action="store_true", help="Recurse into subfolders (rglob)")
    ap.add_argument("--alpha", type=float, default=0.5, help="Alpha for UES (default 0.5)")
    ap.add_argument("--baseline_json", type=Path, default=None,
                    help="Optional JSON with global baselines per concept")
    ap.add_argument("--concept_keys", type=str, default=None,
                    help="Comma-separated concept keys to search in file paths; default = keys of baseline map")
    ap.add_argument("--debug", action="store_true", help="Print files found and parse counts")
    args = ap.parse_args()

    # Load/prepare global baselines
    if args.baseline_json and args.baseline_json.exists():
        with open(args.baseline_json, "r") as f:
            global_baselines = json.load(f)
    else:
        global_baselines = DEFAULT_GLOBAL_BASELINES

    if args.concept_keys:
        concept_keys = [x.strip() for x in args.concept_keys.split(",") if x.strip()]
    else:
        concept_keys = list(global_baselines.keys())

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if not methods:
        print("No methods specified.")
        return

    for method in methods:
        token = method.lower()
        file_iter = args.root.rglob("*.txt") if args.recurse else args.root.glob("*.txt")
        files = []
        for p in sorted(file_iter):
            name = p.name.lower()
            if re.search(rf"(?:^|[^a-z0-9]){re.escape(token)}(?:[^a-z0-9]|$)", name):
                files.append(p)

        if args.debug:
            print(f"[{method}] files found: {len(files)}")
            for p in files[:10]:
                print("   -", p)

        rows = []
        parsed_lines_total = 0

        for fp in files:
            parsed = parse_file(fp)
            parsed_lines_total += len(parsed.get('iter', []))
            sel_idx = select_row(parsed)
            if sel_idx is None:
                continue

            # selected row metrics
            selected = _row_at(parsed, sel_idx)

            # baseline: prefer GLOBAL baseline by concept; else per-file baseline row
            concept = infer_concept_from_path(fp, concept_keys)
            if concept and concept in global_baselines:
                base_map = global_baselines[concept]
                base_row = {
                    'fgt_acc@1': base_map['f1'], 'fgt_acc@5': base_map['f5'],
                    'test_acc@1': base_map['t1'], 'test_acc@5': base_map['t5'],
                    'celeba100@1': base_map['c1'], 'celeba100@5': base_map['c5'],
                }
            else:
                base_idx = _baseline_index(parsed)
                base_row = _row_at(parsed, base_idx)

            # UES for this file
            try:
                ues = _ues_score(
                    f1 = selected.get('fgt_acc@1'), f5 = selected.get('fgt_acc@5'),
                    t1 = selected.get('test_acc@1'), t5 = selected.get('test_acc@5'),
                    c1 = selected.get('celeba100@1'), c5 = selected.get('celeba100@5'),
                    base_f1 = base_row.get('fgt_acc@1'), base_f5 = base_row.get('fgt_acc@5'),
                    base_t1 = base_row.get('test_acc@1'), base_t5 = base_row.get('test_acc@5'),
                    base_c1 = base_row.get('celeba100@1'), base_c5 = base_row.get('celeba100@5'),
                    alpha = args.alpha
                )
            except Exception:
                ues = None

            selected['UES'] = ues
            rows.append(selected)

        if args.debug:
            print(f"[{method}] parsed line-count: {parsed_lines_total}, selected rows: {len(rows)}")

        if not rows:
            print(f"{method}: no usable rows found")
            continue

        avgs = calculate_averages(rows)

        # Build output
        out = [f"{method}:"]
        def add_pct(label):
            if label in avgs and isinstance(avgs[label], (int, float)):
                out.append(f"{label}: {avgs[label]*100:.2f}%")

        for k in ['fgt_acc@1','fgt_acc@5','celeba100@1','celeba100@5','test_acc@1','test_acc@5','nbr@1','nbr@5','TISI','PAR','EGR']:
            add_pct(k)

        if 'MIA' in avgs and isinstance(avgs['MIA'], tuple):
            m, s = avgs['MIA']
            if isinstance(m, (int,float)) and isinstance(s, (int,float)):
                out.append(f"MIA: {m:.2f} ± {s:.2f}")

        if 'UES' in avgs and isinstance(avgs['UES'], (int, float)):
            out.append(f"UES: {avgs['UES']:.4f}")

        out.append(f"(files={len(rows)}, baseline={'GLOBAL+fallback' if concept_keys else 'per-file'})")
        print(", ".join(out))


if __name__ == "__main__":
    main()
