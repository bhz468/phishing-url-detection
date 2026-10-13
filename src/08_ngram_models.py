import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

from features import build_matrix, decode_host, host_of

rng = np.random.default_rng(0)

# --- same data as 04_cross_source.py ---
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
y_te = np.array([1] * len(op) + [0] * len(legit))
y_tr = tr["y"].values
print(f"train {len(tr)} | test {len(op)} phishing + {len(legit)} legitimate")

# --- features ---
Xh_tr = build_matrix(tr["host"].tolist(), top_tlds)
Xh_te = build_matrix(hosts_te, top_tlds)
txt_tr = ["^" + decode_host(h) + "$" for h in tr["host"]]
txt_te = ["^" + decode_host(h) + "$" for h in hosts_te]
vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=5,
                      sublinear_tf=True, max_features=150000, lowercase=False)
T_tr = vec.fit_transform(txt_tr)
T_te = vec.transform(txt_te)
sc = StandardScaler().fit(Xh_tr)
C_tr = hstack([T_tr, csr_matrix(sc.transform(Xh_tr))]).tocsr()
C_te = hstack([T_te, csr_matrix(sc.transform(Xh_te))]).tocsr()

# --- models ---
preds = {}
m = HistGradientBoostingClassifier(random_state=0).fit(Xh_tr, y_tr)
preds["hand features (GB)"] = m.predict_proba(Xh_te)[:, 1]
m_ng = LogisticRegression(max_iter=1000).fit(T_tr, y_tr)
preds["char n-grams (LR)"] = m_ng.predict_proba(T_te)[:, 1]
m_cb = LogisticRegression(max_iter=1000).fit(C_tr, y_tr)
preds["n-grams + hand (LR)"] = m_cb.predict_proba(C_te)[:, 1]


def auc_tpr(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return roc_auc_score(y, p), tpr[np.searchsorted(fpr, 0.05, side="right") - 1]


def evaluate(title, mask):
    y = y_te[mask]
    P = {n: p[mask] for n, p in preds.items()}
    boots = {n: [] for n in P}
    for _ in range(300):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        for n in P:
            boots[n].append(auc_tpr(y[i], P[n][i]))
    base = np.array(boots["hand features (GB)"])
    print(f"\n{title}: {int(y.sum())} phishing, {int((1 - y).sum())} legitimate")
    for n in P:
        b = np.array(boots[n])
        a, t = auc_tpr(y, P[n])
        la, ha = np.percentile(b[:, 0], [2.5, 97.5])
        lt, ht = np.percentile(b[:, 1], [2.5, 97.5])
        line = (f"  {n:<22} AUC {a:.3f} [{la:.3f}-{ha:.3f}] | "
                f"recall@5%FPR {t:.3f} [{lt:.3f}-{ht:.3f}]")
        if n != "hand features (GB)":
            d = b[:, 0] - base[:, 0]
            line += f" | dAUC vs hand {np.mean(d):+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]"
        print(line)


evaluate("ALL TEST HOSTS", np.ones(len(y_te), dtype=bool))
evaluate("BARE DOMAINS ONLY", (Xh_te["n_dots"] == 1).values)

names = np.array(vec.get_feature_names_out())
order = np.argsort(m_ng.coef_[0])
print("\nN-grams the model links most to LEGITIMATE (text only):", list(names[order[:15]]))
print("N-grams the model links most to PHISHING   (text only):", list(names[order[-15:]]))
