from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

OUT = Path("data/phiusiil.csv")


def main():
    OUT.parent.mkdir(exist_ok=True)
    ds = fetch_ucirepo(id=967)  # PhiUSIIL Phishing URL dataset
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    df.to_csv(OUT, index=False)

    label = ds.data.targets.columns[0]
    print(f"Saved {OUT}  shape={df.shape}")
    print("Columns:", list(df.columns))
    print("\nClass counts (UCI page: 1 = legitimate, 0 = phishing):")
    print(df[label].value_counts())

    if "URL" in df.columns:
        dup_urls = df.duplicated(subset=["URL"]).sum()
        print(f"\nDuplicate URLs: {dup_urls}")
    if "Domain" in df.columns:
        per_domain = df.groupby("Domain")[label].nunique()
        print(f"Unique domains: {df['Domain'].nunique()}")
        print(f"Domains present in BOTH classes: {(per_domain > 1).sum()}")
        print("Most frequent domains:")
        print(df["Domain"].value_counts().head(8))


if __name__ == "__main__":
    main()
