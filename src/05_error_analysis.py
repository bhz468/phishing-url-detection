import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from features import build_matrix, host_of

tr = pd.read_csv("data/phiusiil.csv", usecols=["URL", "label"]).drop_duplicates(subset=["URL"])
tr["y"] = 1 - tr["label"]
tr["host"] = tr["URL"].map(host_of)
tr = tr[tr["host"] != ""].reset_index(drop=True)
train_hosts = set(tr["host"])
top_tlds = tr["host"].map(lambda h: h.split(".")[-1]).value_counts().head(30).index.tolist()

with open("data/external/openphish.txt") as f:
    op = pd.Series([host_of(l.strip()) for l in f if l.strip()])
op = op[op != ""].drop_duplicates()
op = op[~op.isin(train_hosts)]

tc = pd.read_csv("data/external/top-1m.csv", header=None, names=["rank", "domain"])
tc = tc[~tc["domain"].str.lower().isin(train_hosts)]
legit = tc.sample(3000, random_state=0)

hosts = op.tolist() + legit["domain"].str.lower().tolist()
y = np.array([1] * len(op) + [0] * len(legit))
rank = [np.nan] * len(op) + legit["rank"].tolist()

X_tr = build_matrix(tr["host"].tolist(), top_tlds)
X_te = build_matrix(hosts, top_tlds)
m = HistGradientBoostingClassifier(random_state=0).fit(X_tr, tr["y"])
res = pd.DataFrame({"host": hosts, "y": y, "p": m.predict_proba(X_te)[:, 1],
                    "rank": rank, "n_dots": X_te["n_dots"].values})
res["dots"] = res["n_dots"].clip(upper=4)

pd.set_option("display.width", 140)
print("LEGITIMATE hosts scored most phishing-like (false positives):")
print(res[res.y == 0].sort_values("p", ascending=False).head(25)[["host", "p", "rank"]]
      .to_string(index=False))
print("\nPHISHING hosts scored most legitimate-like (misses):")
print(res[res.y == 1].sort_values("p").head(25)[["host", "p"]].to_string(index=False))

print("\nShare of hosts by number of dots (4 = four or more):")
print(pd.crosstab(res.y.map({0: "legitimate", 1: "phishing"}), res.dots, normalize="index").round(3))
print("\nFalse-positive rate among legitimate hosts, by number of dots:")
leg = res[res.y == 0]
print(leg.groupby("dots")["p"].agg(fpr=lambda s: (s >= 0.5).mean(), n="count").round(3))
