import pandas as pd
from urllib.parse import urlparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/phiusiil.csv").drop_duplicates(subset=["URL"])
df["is_phishing"] = 1 - df["label"]  # dataset: 1 = legitimate, 0 = phishing

parts = df["URL"].map(urlparse)
df["is_https"] = (parts.map(lambda u: u.scheme) == "https").astype(int)
df["has_path"] = parts.map(lambda u: len(u.path.strip("/")) > 0).astype(int)
df["has_query"] = parts.map(lambda u: bool(u.query)).astype(int)
df["has_www"] = parts.map(lambda u: u.netloc.startswith("www.")).astype(int)
df["len_url"] = df["URL"].str.len()
df["dots_in_host"] = parts.map(lambda u: u.netloc.count("."))
df["tld"] = parts.map(lambda u: u.netloc.split(".")[-1].lower())

cols = ["is_https", "has_path", "has_query", "has_www", "len_url", "dots_in_host"]
print("Mean of simple URL properties (0 = legitimate, 1 = phishing):")
print(df.groupby("is_phishing")[cols].mean().round(3).T)

for cls, name in [(0, "LEGITIMATE"), (1, "PHISHING")]:
    print(f"\nTop 8 TLDs, {name}:")
    print((df[df.is_phishing == cls]["tld"].value_counts(normalize=True).head(8) * 100).round(1))

print("\nSingle trivial features, AUC:")
for c in cols:
    a = roc_auc_score(df["is_phishing"], df[c])
    print(f"  {max(a, 1 - a):.3f}  {c}")

X_tr, X_te, y_tr, y_te = train_test_split(
    df[cols], df["is_phishing"], test_size=0.2, random_state=0, stratify=df["is_phishing"])
model = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
print(f"\nLogistic regression on these 6 trivial features: AUC = "
      f"{roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]):.3f}")

print("\nRandom example URLs (text only, DO NOT open them):")
for cls, name in [(0, "LEGITIMATE"), (1, "PHISHING")]:
    print(f"-- {name}")
    for u in df[df.is_phishing == cls]["URL"].sample(8, random_state=1):
        print("  ", u[:90])
