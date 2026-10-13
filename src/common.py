import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from features import build_matrix, decode_host, host_of


def load_and_fit(n_legit=3000):
    """Train n-grams + hand features (LR) on PhiUSIIL; score OpenPhish + Tranco."""
    tr = pd.read_csv("data/phiusiil.csv", usecols=["URL", "label"]).drop_duplicates(subset=["URL"])
    tr["y"] = 1 - tr["label"]
    tr["host"] = tr["URL"].map(host_of)
    tr = tr[tr["host"] != ""].reset_index(drop=True)
    train_hosts = set(tr["host"])
    top_tlds = tr["host"].map(lambda h: h.split(".")[-1]).value_counts().head(30).index.tolist()

    with open("data/external/merged_openphish.txt") as f:
        op = pd.Series([host_of(l.strip()) for l in f if l.strip()])
    op = op[(op != "") & ~op.isin(train_hosts)].drop_duplicates().tolist()
    tc = pd.read_csv("data/external/tranco/top-1m.csv", header=None, names=["rank", "domain"])
    tc["host"] = tc["domain"].str.lower()
    legit = tc[~tc["host"].isin(train_hosts)]["host"].sample(n_legit, random_state=0).tolist()

    hosts_te = op + legit
    Xh_tr = build_matrix(tr["host"].tolist(), top_tlds)
    Xh_te = build_matrix(hosts_te, top_tlds)
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=5,
                          sublinear_tf=True, max_features=150000, lowercase=False)
    T_tr = vec.fit_transform(["^" + decode_host(h) + "$" for h in tr["host"]])
    T_te = vec.transform(["^" + decode_host(h) + "$" for h in hosts_te])
    sc = StandardScaler().fit(Xh_tr)
    C_tr = hstack([T_tr, csr_matrix(sc.transform(Xh_tr))]).tocsr()
    C_te = hstack([T_te, csr_matrix(sc.transform(Xh_te))]).tocsr()
    m = LogisticRegression(max_iter=1000).fit(C_tr, tr["y"].values)
    p = m.predict_proba(C_te)[:, 1]
    n = len(op)
    return {
        "train": tr, "op": op, "legit": legit,
        "p_op": p[:n], "p_leg": p[n:],
        "dots_op": Xh_te["n_dots"].values[:n], "dots_leg": Xh_te["n_dots"].values[n:],
    }
