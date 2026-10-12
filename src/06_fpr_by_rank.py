import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from features import build_matrix, host_of

tr = pd.read_csv("data/phiusiil.csv", usecols=["URL", "label"]).drop_duplicates(subset=["URL"])
tr["y"] = 1 - tr["label"]
tr["host"] = tr["URL"].map(host_of)
tr = tr[tr["host"] != ""].reset_index(drop=True)
train_hosts = set(tr["host"])
top_tlds = tr["host"].map(lambda h: h.split(".")[-1]).value_counts().head(30).index.tolist()

tc = pd.read_csv("data/external/top-1m.csv", header=None, names=["rank", "domain"])
tc["host"] = tc["domain"].str.lower()
tc = tc[~tc["host"].isin(train_hosts)]

model = HistGradientBoostingClassifier(random_state=0).fit(
    build_matrix(tr["host"].tolist(), top_tlds), tr["y"])


def flagged(hosts):
    return model.predict_proba(build_matrix(list(hosts), top_tlds))[:, 1] >= 0.5


print("False-positive rate by Tranco popularity band (1000 random hosts each):")
for name, lo, hi in [("rank 1 - 10k", 1, 10_000), ("rank 10k - 100k", 10_001, 100_000),
                     ("rank 100k - 1M", 100_001, 1_000_000)]:
    band = tc[(tc["rank"] >= lo) & (tc["rank"] <= hi)]
    s = band.sample(min(1000, len(band)), random_state=0)
    f = flagged(s["host"])
    print(f"  {name:16s} n={len(s)}  FPR={f.mean():.3f}")
    if hi == 1_000_000:
        tlds = s["host"][f].map(lambda h: h.split(".")[-1]).value_counts().head(8)
        print("    TLDs among the flagged hosts in this band:", dict(tlds))

pun = tc[tc["host"].str.contains("xn--")]
s = pun.sample(min(1000, len(pun)), random_state=0)
print(f"\nPunycode (xn--) hosts in Tranco: {len(pun)} available, {len(s)} tested")
print(f"  FPR on punycode hosts: {flagged(s['host']).mean():.3f}")
print(f"  median rank of those hosts: {int(s['rank'].median())}")
