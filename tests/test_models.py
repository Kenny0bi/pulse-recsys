"""Model correctness without the 25M download: gradcheck and logQ direction."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pulse.models import BPRMF, TwoTower


def test_twotower_gradcheck():
    rng = np.random.default_rng(0)
    B, dim, temp = 12, 8, 0.07
    U = rng.standard_normal((B, dim)) * 0.1
    V = rng.standard_normal((B, dim)) * 0.1

    def norm(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

    def loss_of(U_):
        u, v = norm(U_), norm(V)
        logits = (u @ v.T) / temp
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        p = e / e.sum(axis=1, keepdims=True)
        return -np.log(np.diag(p) + 1e-12).mean()

    u, v = norm(U), norm(V)
    logits = (u @ v.T) / temp
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    p = e / e.sum(axis=1, keepdims=True)
    d = (p - np.eye(B)) / (B * temp)
    du = d @ v

    def denorm(g, raw):
        n = np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8
        unit = raw / n
        return (g - (g * unit).sum(axis=1, keepdims=True) * unit) / n

    gU = denorm(du, U)
    eps, worst = 1e-6, 0.0
    for r in range(4):
        for c in range(5):
            U2 = U.copy(); U2[r, c] += eps; up = loss_of(U2)
            U2[r, c] -= 2 * eps; dn = loss_of(U2)
            num = (up - dn) / (2 * eps)
            worst = max(worst, abs(num - gU[r, c]))
    assert worst < 1e-5, f"gradcheck failed: {worst}"
    print(f"  ok  two-tower gradcheck (float64, max err {worst:.2e})")


def test_logq_protects_popular_items():
    """Train two tiny towers on a popularity-skewed world; the corrected one
    must rank the popular item HIGHER than the uncorrected one does."""
    rng = np.random.default_rng(3)
    n_users, n_items = 200, 20
    freq = np.ones(n_items); freq[0] = 60.0     # item 0 dominates
    probs = freq / freq.sum()
    users = rng.integers(0, n_users, 6000)
    items = rng.choice(n_items, 6000, p=probs)

    ranks = {}
    for use_logq in [False, True]:
        tt = TwoTower(n_users, n_items, dim=16, lr=0.05, seed=0,
                      item_freq=freq if use_logq else None)
        for s in range(0, 5800, 64):
            u, i = users[s:s + 64], items[s:s + 64]
            _, first = np.unique(i, return_index=True)
            tt.train_step(u[first], i[first])
        scores = tt.user_vectors(np.arange(50)) @ tt.item_vectors().T
        ranks[use_logq] = float((scores.argsort(axis=1)[:, ::-1] == 0
                                 ).argmax(axis=1).mean())
    assert ranks[True] < ranks[False], (
        f"logQ did not protect the popular item: {ranks}")
    print(f"  ok  mean rank of the popular item: {ranks[False]:.1f} uncorrected "
          f"-> {ranks[True]:.1f} with logQ")


if __name__ == "__main__":
    test_twotower_gradcheck()
    test_logq_protects_popular_items()
    print("all model tests passed")
