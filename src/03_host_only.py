import math
import re
from urllib.parse import urlparse

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/phiusiil.csv", usecols=["URL", "label"]).drop_duplicates(subset=["URL"])
df["is_phishing"] = 1 - df["label"]  # dataset: 1 = legitimate, 0 = phishing


def host_of(url):
    try:
        h = urlparse(url).netloc.lower()
    except ValueError:
        return ""
    h = h.split("@")[-1].split(":")[0]
    return h[4:] if h.startswith("www.") else h


def entropy(s):
    if not s:
        return 0.0
    return -sum((s.count(c) / len(s)) * math.log2(s.count(c) / len(s)) for c in set(s))


df["host"] = df["URL"].map(host_of)
df = df[df["host"] != ""].reset_index(drop=True)


def features(h):
    labels = h.split(".")
    digits = sum(c.isdigit() for c in h)
    return {
        "host_len": len(h),
        "n_dots": h.count("."),
        "n_digits": digits,
        "digit_ratio": digits / len(h),
        "n_hyphens": h.count("-"),
        "entropy": entropy(h.replace(".", "")),
        "longest_label": max(len(x) for x in labels),
        "is_ip": int(bool(re.fullmatch(r"[\d.]+", h))),
        "has_punycode": int("xn--" in h),
        "tld_len": len(labels[-1]),
    }


X = pd.DataFrame([features(h) for h in df["host"]])
tld = df["host"].map(lambda h: h.split(".")[-1])
top = tld.value_counts().head(30).index
tld_cols = pd.get_dummies(tld.where(tld.isin(top), "other"), prefix="tld").astype(int)
y = df["is_phishing"]

split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
tr, te = next(split.split(X, y, groups=df["host"]))
print(f"Rows: {len(df)} | train {len(tr)} | test {len(te)} | phishing share {y.mean():.3f}")


def evaluate(name, Xm):
    models = {
        "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "gradient boosting": HistGradientBoostingClassifier(random_state=0),
    }
    for mname, m in models.items():
        m.fit(Xm.iloc[tr], y.iloc[tr])
        p = m.predict_proba(Xm.iloc[te])[:, 1]
        tn, fp, fn, tp = confusion_matrix(y.iloc[te], p >= 0.5).ravel()
        print(f"{name:26s} {mname:20s} AUC={roc_auc_score(y.iloc[te], p):.3f} "
              f"precision={tp / (tp + fp):.3f} recall={tp / (tp + fn):.3f} "
              f"false-positive-rate={fp / (fp + tn):.3f}")


print("\nHost-only features (scheme, www, path and query removed):")
evaluate("without TLD", X)
evaluate("with TLD", pd.concat([X, tld_cols], axis=1))
