"""Offline retrieval evaluation: recall@K and NDCG@K on the temporal test.

Seen-item filtering matters: recommending something the user already rated
in TRAIN is not a win, so train items are masked out before ranking.
"""

import numpy as np


def build_user_pos(test_u, test_i):
    pos = {}
    for u, i in zip(test_u, test_i):
        pos.setdefault(int(u), set()).add(int(i))
    return pos


def build_seen(train_u, train_i):
    seen = {}
    for u, i in zip(train_u, train_i):
        seen.setdefault(int(u), []).append(int(i))
    return {u: np.array(v) for u, v in seen.items()}


def evaluate_scores(score_fn, users, user_pos, seen, n_items, ks=(10, 50),
                    chunk=2048):
    """score_fn(user_batch) -> (B, n_items) score matrix."""
    recalls = {k: [] for k in ks}
    ndcgs = {k: [] for k in ks}
    kmax = max(ks)
    for s in range(0, len(users), chunk):
        batch = users[s : s + chunk]
        scores = score_fn(batch)
        for row, u in enumerate(batch):
            sc = scores[row]
            if u in seen:
                sc[seen[u]] = -np.inf
            top = np.argpartition(-sc, kmax)[:kmax]
            top = top[np.argsort(-sc[top])]
            gold = user_pos[u]
            for k in ks:
                hits = [1 if t in gold else 0 for t in top[:k]]
                recalls[k].append(sum(hits) / min(len(gold), k))
                dcg = sum(h / np.log2(r + 2) for r, h in enumerate(hits))
                idcg = sum(1 / np.log2(r + 2) for r in range(min(len(gold), k)))
                ndcgs[k].append(dcg / idcg if idcg > 0 else 0.0)
    return ({k: float(np.mean(v)) for k, v in recalls.items()},
            {k: float(np.mean(v)) for k, v in ndcgs.items()})
