import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from features import build_matrix, host_of


def decode_host(h):
    out = []
    for label in h.split("."):
        if label.startswith("xn--"):
            try:
                label = label[4:].encode("ascii").decode("punycode")
            except Exception:
                pass
        out.append(label)
    return ".".join(out)


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


tr = pd.read_csv("data/phiusiil.csv", usecols=["URL", "label"]).drop_duplicates(subset=["URL"])
tr["y"] = 1 - tr["label"]
tr["host"] = tr["URL"].map(host_of)
tr = tr[tr["host"] != ""].reset_index(drop=True)
train_hosts = set(tr["host"])

pun_tr = tr[tr["host"].str.contains("xn--")]
print("Punycode (xn--) hosts in PhiUSIIL training data, by class:")
print(pun_tr["y"].value_counts().rename({0: "legitimate", 1: "phishing"}).to_string())
print(f"  share of phishing rows: {(pun_tr['y'] == 1).sum() / (tr['y'] == 1).sum():.4f} | "
      f"share of legitimate rows: {(pun_tr['y'] == 0).sum() / (tr['y'] == 0).sum():.4f}")

tc_all = pd.read_csv("data/external/top-1m.csv", header=None, names=["rank", "domain"])
tc_all["host"] = tc_all["domain"].str.lower()
top10k = tc_all[tc_all["rank"] <= 10_000]
print(f"\nTranco top-10k hosts excluded because they appear in PhiUSIIL: "
      f"{top10k['host'].isin(train_hosts).sum()} of {len(top10k)}")

tc = tc_all[~tc_all["host"].isin(train_hosts)]
legit = tc["host"].sample(3000, random_state=0).tolist()
pun_pool = tc[tc["host"].str.contains("xn--")]["host"]
pun = pun_pool.sample(min(1000, len(pun_pool)), random_state=0).tolist()
with open("data/external/openphish.txt") as f:
    op = pd.Series([host_of(l.strip()) for l in f if l.strip()])
op = op[(op != "") & ~op.isin(train_hosts)].drop_duplicates().tolist()


def run(name, f_tr, f_op, f_leg, f_pun, drop_flag):
    top = pd.Series([h.split(".")[-1] for h in f_tr]).value_counts().head(30).index.tolist()
    Xtr = build_matrix(f_tr, top)
    Xop, Xleg, Xpun = (build_matrix(h, top) for h in (f_op, f_leg, f_pun))
    if drop_flag:
        Xtr, Xop, Xleg, Xpun = (x.drop(columns=["has_punycode"]) for x in (Xtr, Xop, Xleg, Xpun))
    m = HistGradientBoostingClassifier(random_state=0).fit(Xtr, tr["y"])
    p_op, p_leg, p_pun = (m.predict_proba(x)[:, 1] for x in (Xop, Xleg, Xpun))

    rec, fpr, fpp = (p_op >= .5), (p_leg >= .5), (p_pun >= .5)
    b_op = (Xop["n_dots"] == 1).values
    b_leg = (Xleg["n_dots"] == 1).values
    yb = np.r_[np.ones(b_op.sum()), np.zeros(b_leg.sum())]
    pb = np.r_[p_op[b_op], p_leg[b_leg]]
    r, f, q = wilson(rec.sum(), len(rec)), wilson(fpr.sum(), len(fpr)), wilson(fpp.sum(), len(fpp))
    print(f"\n{name}")
    print(f"  recall on OpenPhish        {rec.mean():.3f} [{r[0]:.3f}-{r[1]:.3f}]  n={len(rec)}")
    print(f"  FPR on Tranco sample       {fpr.mean():.3f} [{f[0]:.3f}-{f[1]:.3f}]  n={len(fpr)}")
    print(f"  FPR on punycode hosts      {fpp.mean():.3f} [{q[0]:.3f}-{q[1]:.3f}]  n={len(fpp)}")
    print(f"  AUC, bare domains only     {roc_auc_score(yb, pb):.3f}  "
          f"({int(b_op.sum())} phishing, {int(b_leg.sum())} legitimate)")


h_tr = tr["host"].tolist()
run("Original (has_punycode kept)", h_tr, op, legit, pun, drop_flag=False)
run("A: has_punycode removed", h_tr, op, legit, pun, drop_flag=True)
dec = lambda hs: [decode_host(h) for h in hs]
run("B: IDN decoded + has_punycode removed", dec(h_tr), dec(op), dec(legit), dec(pun), drop_flag=True)
