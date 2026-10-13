import numpy as np
import pandas as pd
from collections import Counter
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from features import build_matrix, decode_host, host_of

rng = np.random.default_rng(0)

tr = pd.read_csv("data/phiusiil.csv", usecols=["URL", "label"]).drop_duplicates(subset=["URL"])
tr["y"] = 1 - tr["label"]
tr["host"] = tr["URL"].map(host_of)
tr = tr[tr["host"] != ""].reset_index(drop=True)
train_hosts = set(tr["host"])
top_tlds = tr["host"].map(lambda h: h.split(".")[-1]).value_counts().head(30).index.tolist()

with open("data/external/merged_openphish.txt") as f:
    op = pd.Series([host_of(l.strip()) for l in f if l.strip()])
op = op[(op != "") & ~op.isin(train_hosts)].drop_duplicates().tolist()
tc = pd.read_csv("data/external/top-1m.csv", header=None, names=["rank", "domain"])
tc["host"] = tc["domain"].str.lower()
legit = tc[~tc["host"].isin(train_hosts)]["host"].sample(3000, random_state=0).tolist()

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
p_op, p_leg = p[:len(op)], p[len(op):]


def suffix2(h):
    return ".".join(decode_host(h).split(".")[-2:])


train_sfx = set(suffix2(h) for h in tr.loc[tr["y"] == 1, "host"])
seen = np.array([suffix2(h) in train_sfx for h in op])
n_dots = Xh_te["n_dots"].values[:len(op)]
thr = np.quantile(p_leg, 0.95)  # threshold giving 5% false positives on the legitimate sample

print(f"test phishing hosts: {len(op)} | suffix seen in training phishing: {seen.sum()} | not seen: {(~seen).sum()}")
print(f"threshold at 5% FPR on the legitimate sample: {thr:.3f}")
print("Most common suffixes among SEEN test phishing hosts:",
      Counter(suffix2(h) for h, s in zip(op, seen) if s).most_common(8))


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    q = k / n
    d = 1 + z * z / n
    c = (q + z * z / (2 * n)) / d
    h = z * np.sqrt(q * (1 - q) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def auc_ci(p_pos):
    def auc(pp, pl):
        return roc_auc_score(np.r_[np.ones(len(pp)), np.zeros(len(pl))], np.r_[pp, pl])
    b = [auc(p_pos[rng.integers(0, len(p_pos), len(p_pos))],
             p_leg[rng.integers(0, len(p_leg), len(p_leg))]) for _ in range(300)]
    return auc(p_pos, p_leg), np.percentile(b, 2.5), np.percentile(b, 97.5)


for name, mask in [("platform SEEN in training", seen), ("platform NOT seen in training", ~seen)]:
    pp = p_op[mask]
    a, lo, hi = auc_ci(pp)
    k = int((pp >= thr).sum())
    wl, wh = wilson(k, len(pp))
    print(f"\n{name}: {len(pp)} phishing hosts | bare-domain share {np.mean(n_dots[mask] == 1):.2f}")
    print(f"  AUC {a:.3f} [{lo:.3f}-{hi:.3f}] | recall at 5% FPR {k / len(pp):.3f} [{wl:.3f}-{wh:.3f}]")
