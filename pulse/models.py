"""Retrieval models in NumPy with hand-written gradients.

TwoTower: the industry-standard retrieval recipe. A batch of (user, item)
positives; every other item in the batch serves as a negative (in-batch
sampled softmax); both towers are embedding tables here, which is the honest
core of the method (feature towers bolt on later; the training dynamics live
in this loss).

BPRMF: the classic baseline. Bayesian Personalized Ranking with uniformly
sampled negatives: maximize sigma(u.(v_pos - v_neg)).

Both use per-row Adam, because full-table optimizer steps at 160K users
would be almost all wasted work.
"""

import numpy as np


class RowAdam:
    """Adam whose state updates touch only the rows a step used."""

    def __init__(self, shape, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.m = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)
        self.t = np.zeros(shape[0], dtype=np.int64)
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps

    def step(self, table, rows, grad):
        self.t[rows] += 1
        t = self.t[rows][:, None]
        self.m[rows] = self.b1 * self.m[rows] + (1 - self.b1) * grad
        self.v[rows] = self.b2 * self.v[rows] + (1 - self.b2) * grad * grad
        mhat = self.m[rows] / (1 - self.b1 ** t)
        vhat = self.v[rows] / (1 - self.b2 ** t)
        table[rows] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


class TwoTower:
    def __init__(self, n_users, n_items, dim=64, lr=0.01, temp=0.07, seed=0,
                 item_freq=None):
        rng = np.random.default_rng(seed)
        self.U = (rng.standard_normal((n_users, dim)) * 0.1).astype(np.float32)
        self.V = (rng.standard_normal((n_items, dim)) * 0.1).astype(np.float32)
        self.opt_u = RowAdam(self.U.shape, lr=lr)
        self.opt_v = RowAdam(self.V.shape, lr=lr)
        self.temp = temp
        # logQ correction (Yi et al. 2019): popular items appear as in-batch
        # negatives in proportion to their popularity; subtracting log q_j
        # from column j's logit stops training from punishing popularity
        # itself. Without this, the model loses to a popularity counter,
        # which is exactly what happened on the first run.
        if item_freq is not None:
            q = item_freq / item_freq.sum()
            self.logq = np.log(q + 1e-12).astype(np.float32)
        else:
            self.logq = None

    @staticmethod
    def _norm(x):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

    def train_step(self, users, items):
        """In-batch softmax: batch positives on the diagonal. Duplicate items
        inside a batch would make false negatives, so callers dedupe."""
        u_raw, v_raw = self.U[users], self.V[items]
        u, v = self._norm(u_raw), self._norm(v_raw)
        logits = (u @ v.T) / self.temp
        if self.logq is not None:
            logits = logits - self.logq[items][None, :]
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        p = e / e.sum(axis=1, keepdims=True)
        B = len(users)
        loss = -np.log(p[np.arange(B), np.arange(B)] + 1e-12).mean()

        d = (p - np.eye(B, dtype=np.float32)) / (B * self.temp)
        du = d @ v
        dv = d.T @ u
        # back through the L2 normalization
        def denorm(g, raw):
            n = np.linalg.norm(raw, axis=1, keepdims=True) + 1e-8
            unit = raw / n
            return (g - (g * unit).sum(axis=1, keepdims=True) * unit) / n
        self.opt_u.step(self.U, users, denorm(du, u_raw))
        self.opt_v.step(self.V, items, denorm(dv, v_raw))
        return float(loss)

    def item_vectors(self):
        return self._norm(self.V)

    def user_vectors(self, users):
        return self._norm(self.U[users])


class BPRMF:
    def __init__(self, n_users, n_items, dim=64, lr=0.08, reg=1e-5, seed=0):
        rng = np.random.default_rng(seed)
        self.U = (rng.standard_normal((n_users, dim)) * 0.1).astype(np.float32)
        self.V = (rng.standard_normal((n_items, dim)) * 0.1).astype(np.float32)
        self.n_items = n_items
        self.lr, self.reg = lr, reg
        self.rng = np.random.default_rng(seed + 1)

    def train_step(self, users, items):
        neg = self.rng.integers(0, self.n_items, len(items))
        u, vp, vn = self.U[users], self.V[items], self.V[neg]
        x = (u * (vp - vn)).sum(axis=1)
        g = 1.0 / (1.0 + np.exp(x))          # d/dx of -log sigmoid(x), negated
        gu = g[:, None] * (vp - vn) - self.reg * u
        gp = g[:, None] * u - self.reg * vp
        gn = -g[:, None] * u - self.reg * vn
        np.add.at(self.U, users, self.lr * gu)
        np.add.at(self.V, items, self.lr * gp)
        np.add.at(self.V, neg, self.lr * gn)
        return float(-np.log(1 / (1 + np.exp(-x)) + 1e-12).mean())
