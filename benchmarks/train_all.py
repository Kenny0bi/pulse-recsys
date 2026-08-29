"""Train and evaluate the ladder: popularity -> BPR-MF -> two-tower.

Everything logged to assets/results.json; loss curves to JSONL.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from pulse.evaluate import build_seen, build_user_pos, evaluate_scores
from pulse.models import BPRMF, TwoTower

DATA = ROOT / "data"
EPOCHS_TT = 6
EPOCHS_BPR = 4
BATCH = 4096


def batches(tr_u, tr_i, rng, batch):
    """Shuffled batches with in-batch item dedupe (false-negative guard)."""
    order = rng.permutation(len(tr_u))
    for s in range(0, len(order) - batch, batch):
        idx = order[s : s + batch]
        u, i = tr_u[idx], tr_i[idx]
        _, first = np.unique(i, return_index=True)
        yield u[first], i[first]


def main():
    d = np.load(DATA / "prepared.npz")
    tr_u, tr_i = d["train_user"], d["train_item"]
    te_u, te_i = d["test_user"], d["test_item"]
    n_users, n_items = int(d["n_users"]), int(d["n_items"])
    pop = np.load(DATA / "popularity.npy").astype(np.float32)
    print(f"train {len(tr_u):,} | test {len(te_u):,} | "
          f"{n_users:,} users x {n_items:,} items", flush=True)

    user_pos = build_user_pos(te_u, te_i)
    seen = build_seen(tr_u, tr_i)
    eval_users = np.array(sorted(user_pos.keys()))
    print(f"eval users: {len(eval_users):,}", flush=True)
    results = {}

    # ---- popularity baseline
    def pop_scores(batch_users):
        return np.tile(pop, (len(batch_users), 1)).copy()

    r, n = evaluate_scores(pop_scores, eval_users, user_pos, seen, n_items)
    results["popularity"] = {"recall": r, "ndcg": n}
    print("popularity:", results["popularity"], flush=True)

    # ---- BPR-MF
    bpr = BPRMF(n_users, n_items, dim=64)
    rng = np.random.default_rng(0)
    t0 = time.time()
    log = []
    for ep in range(EPOCHS_BPR):
        for step, (u, i) in enumerate(batches(tr_u, tr_i, rng, BATCH)):
            loss = bpr.train_step(u, i)
            if step % 200 == 0:
                log.append({"model": "bpr", "epoch": ep, "step": step, "loss": loss})
                print(f"bpr ep{ep} step{step} loss {loss:.4f} "
                      f"({time.time() - t0:,.0f}s)", flush=True)

    def bpr_scores(batch_users):
        return bpr.U[batch_users] @ bpr.V.T

    r, n = evaluate_scores(bpr_scores, eval_users, user_pos, seen, n_items)
    results["bpr_mf"] = {"recall": r, "ndcg": n, "train_s": round(time.time() - t0, 1)}
    print("bpr_mf:", results["bpr_mf"], flush=True)

    # ---- two-tower with in-batch softmax
    tt = TwoTower(n_users, n_items, dim=64, item_freq=pop)
    rng = np.random.default_rng(1)
    t0 = time.time()
    for ep in range(EPOCHS_TT):
        for step, (u, i) in enumerate(batches(tr_u, tr_i, rng, BATCH)):
            loss = tt.train_step(u, i)
            if step % 200 == 0:
                log.append({"model": "twotower", "epoch": ep, "step": step, "loss": loss})
                print(f"tt ep{ep} step{step} loss {loss:.4f} "
                      f"({time.time() - t0:,.0f}s)", flush=True)

    V = tt.item_vectors()

    def tt_scores(batch_users):
        return tt.user_vectors(batch_users) @ V.T

    r, n = evaluate_scores(tt_scores, eval_users, user_pos, seen, n_items)
    results["two_tower"] = {"recall": r, "ndcg": n, "train_s": round(time.time() - t0, 1)}
    print("two_tower:", results["two_tower"], flush=True)

    np.savez_compressed(DATA / "embeddings.npz", U=tt.U, V=tt.V)
    (ROOT / "assets/results.json").write_text(json.dumps(results, indent=1))
    (ROOT / "assets/losses.jsonl").write_text(
        "\n".join(json.dumps(r) for r in log))
    print("training complete", flush=True)


if __name__ == "__main__":
    main()
