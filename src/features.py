import math
from urllib.parse import urlparse

import pandas as pd


def host_of(url: str) -> str:
    try:
        h = urlparse(url if "://" in url else "http://" + url).netloc.lower()
    except ValueError:
        return ""
    h = h.split("@")[-1].split(":")[0]
    return h[4:] if h.startswith("www.") else h


def entropy(s: str) -> float:
    if not s:
        return 0.0
    return -sum((s.count(c) / len(s)) * math.log2(s.count(c) / len(s)) for c in set(s))


def host_features(h: str) -> dict:
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
        "is_ip": int(all(c.isdigit() or c == "." for c in h)),
        "has_punycode": int("xn--" in h),
        "tld_len": len(labels[-1]),
    }


def build_matrix(hosts, top_tlds):
    """Numeric features plus one-hot TLD with a FIXED column set."""
    X = pd.DataFrame([host_features(h) for h in hosts])
    tld = pd.Series([h.split(".")[-1] for h in hosts])
    tld = tld.where(tld.isin(top_tlds), "other")
    for t in list(top_tlds) + ["other"]:
        X["tld_" + t] = (tld == t).astype(int).values
    return X
