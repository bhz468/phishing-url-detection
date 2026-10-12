import pandas as pd
from sklearn.metrics import roc_auc_score

df = pd.read_csv("data/phiusiil.csv")
df["is_phishing"] = 1 - df["label"]  # dataset: 1 = legitimate, 0 = phishing

# 1. Duplicates and conflicting labels
dups = df[df.duplicated(subset=["URL"], keep=False)]
conflicts = (dups.groupby("URL")["is_phishing"].nunique() > 1).sum()
print(f"Rows in duplicate groups: {len(dups)} | URLs with conflicting labels: {conflicts}")

# 2. How well does each single numeric feature separate the classes?
num = df.select_dtypes("number").drop(columns=["label", "is_phishing"])
rows = []
for col in num.columns:
    if num[col].nunique() > 1:
        auc = roc_auc_score(df["is_phishing"], num[col])
        rows.append((col, round(max(auc, 1 - auc), 3)))
rows.sort(key=lambda r: r[1], reverse=True)

print("\nTop 20 single features by AUC (0.5 = useless, 1.0 = perfect):")
for name, auc in rows[:20]:
    print(f"  {auc:.3f}  {name}")

# 3. Simple facts about the raw URL text
df["len_url"] = df["URL"].str.len()
print("\nMean URL length by class (0 = legit, 1 = phishing):")
print(df.groupby("is_phishing")["len_url"].mean().round(1))
print("\nShare of HTTPS URLs by class:")
print(df.groupby("is_phishing")["URL"].apply(lambda s: s.str.startswith("https").mean().round(3)))
