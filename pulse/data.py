"""MovieLens-25M preparation in DuckDB: the split is the science.

Random splits leak the future into training (you rated the sequel after the
original; a random split may train on the sequel and test on the original).
Everything here is split by TIME: train is every interaction before the
cutoff, test is what each user did after. Items unseen before the cutoff are
dropped from test, because no honest system can recommend what did not exist.
"""

import sys
from pathlib import Path

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

MIN_ITEM_RATINGS = 5
MIN_USER_RATINGS = 5
POSITIVE_THRESHOLD = 4.0     # a rating of 4+ counts as a positive interaction
TEST_FRACTION = 0.1          # last 10% of interactions by global time


def prepare():
    con = duckdb.connect()
    con.execute(f"""
        CREATE VIEW raw AS
        SELECT userId AS user, movieId AS item, rating, timestamp AS ts
        FROM read_csv_auto('{DATA}/ml-25m/ratings.csv')
        WHERE rating >= {POSITIVE_THRESHOLD}
    """)
    n_all = con.execute("SELECT count(*) FROM raw").fetchone()[0]
    cutoff = con.execute(
        f"SELECT quantile_cont(ts, {1 - TEST_FRACTION}) FROM raw").fetchone()[0]

    con.execute(f"""
        CREATE TABLE train_raw AS SELECT * FROM raw WHERE ts < {cutoff};
        CREATE TABLE test_raw  AS SELECT * FROM raw WHERE ts >= {cutoff};
    """)
    # eligibility is decided by TRAIN data only: no peeking past the cutoff
    con.execute(f"""
        CREATE TABLE items AS
        SELECT item, count(*) AS cnt FROM train_raw
        GROUP BY item HAVING count(*) >= {MIN_ITEM_RATINGS};
        CREATE TABLE users AS
        SELECT user, count(*) AS cnt FROM train_raw
        GROUP BY user HAVING count(*) >= {MIN_USER_RATINGS};
    """)
    train = con.execute("""
        SELECT t.user, t.item, t.ts FROM train_raw t
        JOIN items USING (item) JOIN users USING (user)
        ORDER BY t.ts
    """).fetchnumpy()
    test = con.execute("""
        SELECT t.user, t.item FROM test_raw t
        JOIN items USING (item) JOIN users USING (user)
    """).fetchnumpy()

    users = np.unique(train["user"])
    items = np.unique(train["item"])
    u_map = {u: i for i, u in enumerate(users)}
    i_map = {m: i for i, m in enumerate(items)}

    tr_u = np.array([u_map[u] for u in train["user"]], dtype=np.int32)
    tr_i = np.array([i_map[m] for m in train["item"]], dtype=np.int32)
    mask = np.array([u in u_map and m in i_map
                     for u, m in zip(test["user"], test["item"])])
    te_u = np.array([u_map[u] for u in test["user"][mask]], dtype=np.int32)
    te_i = np.array([i_map[m] for m in test["item"][mask]], dtype=np.int32)

    np.savez_compressed(
        DATA / "prepared.npz",
        train_user=tr_u, train_item=tr_i,
        test_user=te_u, test_item=te_i,
        n_users=len(users), n_items=len(items),
        item_ids=items, user_ids=users)
    pop = np.bincount(tr_i, minlength=len(items))
    np.save(DATA / "popularity.npy", pop)
    print(f"raw positives: {n_all:,}")
    print(f"train: {len(tr_u):,} interactions | test: {len(te_u):,}")
    print(f"users: {len(users):,} | items: {len(items):,}")
    print(f"cutoff timestamp: {cutoff:.0f}")


if __name__ == "__main__":
    prepare()
