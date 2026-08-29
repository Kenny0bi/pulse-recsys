"""Candidate serving through hive-index's HNSW: the retrieval stack closed.

The two-tower's item embeddings go into the HNSW built in this series'
sixth project; retrieval quality is measured as recall against the
exact-dot-product top-K (how much the ANN index costs) and end-to-end
recommendation recall (whether users can tell).
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "hive-index"))     # project 6's index

import numpy as np

from hive.hnsw import HNSW
from pulse.evaluate import build_seen, build_user_pos

DATA = ROOT / "data"


def main():
    d = np.load(DATA / "prepared.npz")
    emb = np.load(DATA / "embeddings.npz")
    U, V = emb["U"], emb["V"]
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
    Un = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-8)

    # cosine similarity == L2 on normalized vectors, which hive speaks
    print(f"building HNSW over {len(Vn):,} item embeddings...", flush=True)
    t0 = time.time()
    index = HNSW(dim=Vn.shape[1], M=16, ef_construction=100, seed=0)
    index.add_batch(Vn.astype(np.float32), log_every=10000)
    build_s = time.time() - t0
    print(f"built in {build_s:,.0f}s", flush=True)

    rng = np.random.default_rng(0)
    test_users = rng.choice(len(Un), 500, replace=False)
    out = {"build_s": round(build_s, 1), "n_items": int(len(Vn))}
    for ef in [20, 50, 100]:
        agree, qps_t0 = [], time.perf_counter()
        for u in test_users:
            q = Un[u].astype(np.float32)
            exact = set(np.argsort(-(q @ Vn.T))[:10].tolist())
            approx = {p for p, _ in index.search(q, k=10, ef=ef)}
            agree.append(len(exact & approx) / 10)
        dt = time.perf_counter() - qps_t0
        out[f"ef={ef}"] = {"recall_vs_exact": round(float(np.mean(agree)), 4),
                           "qps": round(len(test_users) / dt, 1)}
        print(ef, out[f"ef={ef}"], flush=True)
    (ROOT / "assets/serve_ann.json").write_text(json.dumps(out, indent=1))
    print("serving benchmark complete", flush=True)


if __name__ == "__main__":
    main()
