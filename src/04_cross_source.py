import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import build_matrix, host_of

N_LEGIT = 3000
rng = np.random.default_rng(0)

# --- training data: PhiUSIIL (1 = legitimate, 0 = phishing in the file) ---
tr = pd.read_csv("data/phiusiil.csv", usecols=["URL", "label"]).drop_duplicates(subset=["URL"])
tr["y"] = 1 - tr["label"]
tr["host"] = tr["URL"].map(host_of)
tr = tr[tr["host"] != ""].reset_index(drop=True)
train_hosts = set(tr["host"])
top_tlds = tr["host"].map(lambda h: h.split(".")[-1]).value_counts().head(30).index.tolist()

# --- test data: OpenPhish (phishing) and Tranco (legitimate) ---
with open("data/external/openphish.txt") as f:
    op_urls = [line.strip() for line in f if line.strip()]
op_hosts = pd.Series([host_of(u) for u in op_urls])
op_hosts = op_hosts[op_hosts != ""].drop_duplicates()
n_op_unique = len(op_hosts)
op_hosts = op_hosts[~op_hosts.isin(train_hosts)]

tc = pd.read_csv("data/external/top-1m.csv", header=None, names=["rank", "domain"])
tc["host"] = tc["domain"].str.lower()
tc = tc[~tc["host"].isin(train_hosts)]
legit_hosts = tc["host"].sample(N_LEGIT, random_state=0)

print(f"Training rows: {len(tr)} (phishing share {tr['y'].mean():.3f})")
print(f"OpenPhish unique hosts: {n_op_unique} -> {len(op_hosts)} after removing hosts seen in training")
print(f"Tranco sample: {len(legit_hosts)} hosts (not seen in training)")

hosts_te = pd.concat([op_hosts, legit_hosts]).tolist()
y_te = np.array([1] * len(op_hosts) + [0] * len(legit_hosts))
X_tr = build_matrix(tr["host"].tolist(), top_tlds)
X_te = build_matrix(hosts_te, top_tlds)


def stats(y, p, idx):
    yy, pp = y[idx], p[idx]
    pred = pp >= 0.5
    return roc_auc_score(yy, pp), pred[yy == 1].mean(), pred[yy == 0].mean()


def report(name, y, p):
    base = stats(y, p, np.arange(len(y)))
    boots = []
    for _ in range(300):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) == 2:
            boots.append(stats(y, p, idx))
    lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
    auc, tpr, fpr = base
    prec = tpr * 0.01 / (tpr * 0.01 + fpr * 0.99) if (tpr + fpr) > 0 else float("nan")
    print(f"  {name:<18} AUC {auc:.3f} [{lo[0]:.3f}-{hi[0]:.3f}] | "
          f"recall {tpr:.3f} [{lo[1]:.3f}-{hi[1]:.3f}] | "
          f"FPR {fpr:.3f} [{lo[2]:.3f}-{hi[2]:.3f}] | precision at 1% prevalence {prec:.2f}")


models = {
    "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "gradient boosting": HistGradientBoostingClassifier(random_state=0),
}
bare = (X_te["n_dots"] == 1).values
print(f"\nBare-domain subset: {int(y_te[bare].sum())} phishing, {int((1 - y_te[bare]).sum())} legitimate")

for mname, m in models.items():
    m.fit(X_tr, tr["y"])
    p = m.predict_proba(X_te)[:, 1]
    print(f"\n{mname} (trained on PhiUSIIL, tested on OpenPhish + Tranco)")
    report("all test hosts", y_te, p)
    if y_te[bare].sum() >= 30 and (1 - y_te[bare]).sum() >= 30:
        report("bare domains only", y_te[bare], p[bare])
    else:
        print("  bare domains only: too few phishing hosts for a stable estimate")
