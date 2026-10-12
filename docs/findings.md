# Findings

All numbers come from the scripts in `src/` (run in order). Phishing test data:
OpenPhish free feed snapshot of 2026-10-12 (230 unique hosts not seen in
training). Legitimate test data: random sample of 3,000 hosts from the Tranco
top-1M list, excluding hosts present in the training set. Raw data is not
redistributed (see .gitignore).

## 1. The training dataset has strong collection artifacts (PhiUSIIL)
- Every legitimate URL is `https://www.<domain>`: 100% HTTPS, 100% `www`, 0% with
  a path or query. Phishing URLs: 48.6% HTTPS, 27.1% with a path.
- A logistic regression on six trivial URL properties reaches AUC 0.997.
- Page-content features (LineOfCode, NoOfExternalRef, ...) separate the classes
  almost perfectly alone (AUC 0.97-0.99). They also require fetching pages, so
  they are not used.

## 2. Host-only features, same-source split
| Model | AUC | Precision | Recall | False-positive rate |
|---|---|---|---|---|
| logistic, no TLD | 0.753 | 0.905 | 0.534 | 0.043 |
| gradient boosting, no TLD | 0.838 | 0.908 | 0.574 | 0.045 |
| logistic, with TLD | 0.862 | 0.913 | 0.604 | 0.045 |
| gradient boosting, with TLD | 0.848 | 0.930 | 0.643 | 0.038 |

Single split, single seed: differences of about 0.01 are not reliable.

## 3. Cross-source test (train on PhiUSIIL, test on OpenPhish + Tranco)
| | all test hosts | bare domains only |
|---|---|---|
| AUC | 0.903-0.905 | 0.633-0.647 |
| Recall | 0.804-0.826 | 0.319-0.340 |
| False-positive rate | 0.165-0.169 | 0.173-0.180 |

Precision at an assumed 1% phishing prevalence (calculated, not measured):
about 0.05 overall, about 0.02 on bare domains. Only 47 phishing hosts are bare
domains, so intervals are wide (see script output).

## 4. Error analysis
- False positives are mostly random-string hosts on cheap TLDs (.top, .xyz, .vip).
  Some "legitimate" Tranco hosts may not be trustworthy, so the false-positive
  rate is probably overestimated.
- Misses include brand impersonation (roblox.com.* variants), URL shorteners and
  plausibly compromised legitimate-looking sites: out of reach for host-only
  lexical features.
- False-positive rate by Tranco band: rank 1-10k 11.1%, 10k-100k 16.0%,
  100k-1M 17.6% (the top band excludes 1,363 hosts that are also in PhiUSIIL).

## 5. A bias found in my own features
- PhiUSIIL contains 154 punycode (`xn--`) hosts, all phishing, none legitimate.
- Result: every one of 1,000 Tranco punycode hosts was flagged (FPR 100%).
- Removing the `has_punycode` flag alone: FPR 98%. Decoding names to Unicode
  and removing the flag: FPR 25.3% [22.7-28.1%], with recall and other metrics
  unchanged.

## Limitations
- One OpenPhish snapshot (230 hosts, one day); the feed lists only active URLs.
- Tranco ranks popularity, not legitimacy.
- Host-level features only; no URL paths (no source with legitimate URLs that
  have paths yet).
- Single train/test split and seed for the same-source results.
