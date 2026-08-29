"""pulse-recsys visuals: a night at the movies. Plum-black ground, marquee
rose and gold, silver text.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

BG = "#171019"
PANEL = "#201826"
GRID = "#392c40"
INK = "#efe8f4"
FAINT = "#94849e"
ROSE = "#fb7185"
GOLD = "#facc15"
TEAL = "#2dd4bf"
RED = "#f87171"


def header(w, h, title, subtitle):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'font-family="Menlo, Consolas, monospace">',
        f'<rect width="{w}" height="{h}" fill="{BG}" rx="14"/>',
        f'<text x="46" y="52" fill="{INK}" font-size="21" font-weight="bold">{title}</text>',
        f'<text x="46" y="74" fill="{FAINT}" font-size="12.5" '
        f'font-family="Georgia, serif" font-style="italic">{subtitle}</text>',
    ]


# ------------------------------------- 1. the ladder, including the failure

def render_ladder():
    res = json.loads((ASSETS / "results.json").read_text())
    rows = [
        ("popularity counter", res["popularity"]["recall"]["10"]
         if "10" in res["popularity"]["recall"] else res["popularity"]["recall"][10], FAINT,
         "no learning at all: recommend what everyone likes"),
        ("BPR matrix factorization", res["bpr_mf"]["recall"]["10"]
         if "10" in res["bpr_mf"]["recall"] else res["bpr_mf"]["recall"][10], TEAL,
         "the 2009 classic, tuned fairly (lr 0.08, 4 epochs)"),
        ("two-tower, first attempt", 0.0181, RED,
         "in-batch softmax without the logQ correction"),
        ("two-tower + logQ", res["two_tower"]["recall"]["10"]
         if "10" in res["two_tower"]["recall"] else res["two_tower"]["recall"][10], GOLD,
         "one line of math: logits minus log item frequency"),
    ]
    W, H = 1160, 520
    x0, bar = 420, 600
    hi = max(v for _, v, _, _ in rows) * 1.12
    s = header(W, H, "counting, learning, and one line of math",
               "recall@10 on the temporal test, 25M MovieLens ratings: the "
               "first two-tower lost to a counter, and the fix is in the caption")
    for i, (name, v, col, desc) in enumerate(rows):
        y = 150 + i * 78
        s.append(f'<text x="{x0 - 16}" y="{y - 2}" fill="{INK}" font-size="13" '
                 f'text-anchor="end">{name}</text>')
        s.append(f'<text x="{x0 - 16}" y="{y + 16}" fill="{FAINT}" font-size="10" '
                 f'text-anchor="end">{desc}</text>')
        w = bar * v / hi
        s.append(f'<line x1="{x0}" y1="{y + 4}" x2="{x0 + w:.0f}" y2="{y + 4}" '
                 f'stroke="{col}" stroke-width="26" stroke-linecap="round"/>')
        s.append(f'<text x="{x0 + w + 16:.0f}" y="{y + 9}" fill="{col}" '
                 f'font-size="14" font-weight="bold">{v:.3f}</text>')
    s.append(f'<text x="70" y="{H - 58}" fill="{FAINT}" font-size="10.5" '
             f'font-family="Georgia, serif" font-style="italic">'
             'the gradients were proven correct (float64 check, 1.3e-06) before the '
             'failure was blamed on the objective:</text>')
    s.append(f'<text x="70" y="{H - 42}" fill="{FAINT}" font-size="10.5" '
             f'font-family="Georgia, serif" font-style="italic">'
             'popular items were being punished for appearing as in-batch negatives '
             'in proportion to their popularity</text>')
    s.append("</svg>")
    (ASSETS / "ladder.svg").write_text("\n".join(s))
    print("ladder.svg")


# --------------------------------------------- 2. the river and the cutoff

def render_timeline():
    t = json.loads((ASSETS / "timeline.json").read_text())
    years = [(y, n) for y, n in t["per_year"] if y >= 1996]
    W, H = 1160, 460
    px0, px1, py1 = 100, W - 80, H - 120
    hi = max(n for _, n in years)
    band = (px1 - px0) / len(years)
    s = header(W, H, "the river of ratings, and where it was cut",
               "positive ratings (4 stars and up) per year; everything after "
               "the cutoff is the future, and the models never see it")
    cutoff_x = None
    for i, (y, n) in enumerate(years):
        x = px0 + band * (i + 0.5)
        h = 240 * n / hi
        col = ROSE if y <= 2017 else GRID
        s.append(f'<line x1="{x:.0f}" y1="{py1}" x2="{x:.0f}" y2="{py1 - h:.0f}" '
                 f'stroke="{col}" stroke-width="{band * 0.62:.0f}"/>')
        if y % 4 == 0:
            s.append(f'<text x="{x:.0f}" y="{py1 + 20}" fill="{FAINT}" '
                     f'font-size="10" text-anchor="middle">{y}</text>')
        if y == 2017:
            cutoff_x = x + band / 2
    s.append(f'<line x1="{cutoff_x:.0f}" y1="{py1 - 270}" x2="{cutoff_x:.0f}" '
             f'y2="{py1 + 8}" stroke="{GOLD}" stroke-width="2.5" '
             f'stroke-dasharray="7 5"/>')
    s.append(f'<text x="{cutoff_x - 12:.0f}" y="{py1 - 250}" fill="{GOLD}" '
             f'font-size="11.5" text-anchor="end">the cutoff: Dec 31, 2017</text>')
    s.append(f'<text x="{cutoff_x - 12:.0f}" y="{py1 - 232}" fill="{FAINT}" '
             f'font-size="10.5" text-anchor="end">train on the past, test on the future</text>')
    s.append(f'<text x="{px0}" y="{H - 40}" fill="{FAINT}" font-size="10.5" '
             f'font-family="Georgia, serif" font-style="italic">'
             'a random split would let the model peek across this line; item '
             'and user eligibility are decided from the rose side only</text>')
    s.append("</svg>")
    (ASSETS / "timeline.svg").write_text("\n".join(s))
    print("timeline.svg")


# -------------------------------------------------- 3. the retrieval handoff

def render_handoff():
    d = json.loads((ASSETS / "serve_ann.json").read_text())
    W, H = 1160, 440
    px0, py1 = 220, H - 130
    s = header(W, H, "the handoff: my embeddings through my index",
               "the two-tower's 17K item vectors served by the HNSW built in "
               "this series' sixth project")
    efs = [k for k in d if k.startswith("ef=")]
    band = 700 / len(efs)
    for i, ef in enumerate(efs):
        x = px0 + band * (i + 0.5)
        r = d[ef]["recall_vs_exact"]
        q = d[ef]["qps"]
        h = 200 * r
        s.append(f'<line x1="{x:.0f}" y1="{py1}" x2="{x:.0f}" y2="{py1 - h:.0f}" '
                 f'stroke="{GOLD}" stroke-width="42" stroke-linecap="round"/>')
        s.append(f'<text x="{x:.0f}" y="{py1 - h - 44:.0f}" fill="{GOLD}" '
                 f'font-size="13" text-anchor="middle" font-weight="bold">{r:.3f}</text>')
        s.append(f'<text x="{x:.0f}" y="{py1 - h - 28:.0f}" fill="{TEAL}" '
                 f'font-size="10.5" text-anchor="middle">{q:,.0f} qps</text>')
        s.append(f'<text x="{x:.0f}" y="{py1 + 34}" fill="{INK}" font-size="11.5" '
                 f'text-anchor="middle">{ef}</text>')
    s.append(f'<text x="{px0}" y="{H - 44}" fill="{FAINT}" font-size="10.5" '
             f'font-family="Georgia, serif" font-style="italic">'
             'recall vs the exact dot-product top-10 · the index was built in '
             f'{d["build_s"]:.0f}s · projects composing is the point of a series</text>')
    s.append("</svg>")
    (ASSETS / "handoff.svg").write_text("\n".join(s))
    print("handoff.svg")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in {"ladder": render_ladder, "timeline": render_timeline,
                     "handoff": render_handoff}.items():
        if which in ("all", name):
            try:
                fn()
            except FileNotFoundError as e:
                print(f"skip {name}: {e}")
