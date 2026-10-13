from collections import Counter

import numpy as np
from sklearn.metrics import roc_auc_score

from common import load_and_fit
from features import decode_host

rng = np.random.default_rng(0)
d = load_and_fit()
op, legit, tr = d["op"], d["legit"], d["train"]


def suffix2(h):
    return ".".join(decode_host(h).split(".")[-2:])


train_sfx = set(suffix2(h) for h in tr.loc[tr["y"] == 1, "host"])
seen = np.array([suffix2(h) in train_sfx for h in op])
p_op, p_leg = d["p_op"], d["p_leg"]
op_bare, leg_bare = d["dots_op"] == 1, d["dots_leg"] == 1


def auc(pp, pl):
    return roc_auc_score(np.r_[np.ones(len(pp)), np.zeros(len(pl))], np.r_[pp, pl])


def row(name, pp, pl):
    if len(pp) < 20 or len(pl) < 20:
        print(f"  {name:<44} too few hosts ({len(pp)} phishing, {len(pl)} legitimate)")
        return
    b = [auc(pp[rng.integers(0, len(pp), len(pp))], pl[rng.integers(0, len(pl), len(pl))])
         for _ in range(300)]
    print(f"  {name:<44} {len(pp):>3} phishing vs {len(pl):>4} legit | "
          f"AUC {auc(pp, pl):.3f} [{np.percentile(b, 2.5):.3f}-{np.percentile(b, 97.5):.3f}]")


print("Hosts WITH subdomains (n_dots >= 2):")
row("phishing on SEEN platforms", p_op[seen & ~op_bare], p_leg[~leg_bare])
row("phishing on UNSEEN platforms", p_op[~seen & ~op_bare], p_leg[~leg_bare])
print("\nBare domains (n_dots == 1):")
row("phishing, platform SEEN (suffix in training)", p_op[seen & op_bare], p_leg[leg_bare])
row("phishing, platform UNSEEN", p_op[~seen & op_bare], p_leg[leg_bare])

print("\nLegitimate hosts with subdomains in the Tranco sample:", int((~leg_bare).sum()))
subs = Counter(suffix2(h) for h, b in zip(legit, leg_bare) if not b)
print("  most common suffixes:", subs.most_common(8))
