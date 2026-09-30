[ 🇺🇸 English ] | [ 🇨🇱 [Leer en Español](README.es.md) ]

# Fraud Detection Techniques Lab

[![tests](https://github.com/Rxyxs/fraud-detection-techniques-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/Rxyxs/fraud-detection-techniques-lab/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![Techniques](https://img.shields.io/badge/techniques-4%20self--contained-8A2BE2)
![Data](https://img.shields.io/badge/data-2%20real%20datasets%20%2B%202%20synthetic-4479A1)
![Languages](https://img.shields.io/badge/languages-Python%20·%20R%20·%20Julia%20·%20Rust%20·%20SQL-000000)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Four fraud/AML detection approaches in one lab, each isolating a different technique on a different data profile — real-world card fraud, real-time e-commerce, interbank AML typologies, and a direct autoencoder-vs-supervised comparison. Each folder is self-contained with its own README, dependencies, tests and figures.

---

## What the four techniques found

Every number below comes from an actual run of that folder's pipeline, not from a previous version or a cited benchmark.

| # | Technique | Headline result | The finding worth reading |
|---|---|---|---|
| **01** | [Multi-language pipeline on real data](01-credit-card-fraud-multilang) | Business cost **260 → 180** after Optuna tuning | PR-AUC was already saturated at 1.0000 and went *down* (0.999988). **Tuning paid off in the cost-calibrated threshold, not the ranking metric** — reported exactly as it happened |
| **02** | [Real-time detection](02-realtime-ecommerce-fraud) | Recall **0.955** at precision **1.000** — 105/110 frauds caught, **0 false alarms**, scored at **1.59ms p95** | A real architecture bug: dying ReLU in a 4-unit autoencoder bottleneck — negative pre-activations starved a network with no spare capacity. LeakyReLU lifted recall 0.900 → 0.955 (11 missed frauds → 5) and F1 0.947 → 0.977 |
| **03** | [Graph-based unsupervised AML](03-graph-based-aml-detection) | ROC-AUC **0.893** with **zero labels**, on 43,009 transfers | Precision peaks at a **3% alert budget (51.7%), not 1% (40.0%)** — the very top of the ranking is a handful of extreme outliers, and widening slightly finds more true positives |
| **04** | [Autoencoder vs. supervised](04-autoencoder-vs-supervised) | Cold-start → mature: PR-AUC **0.242 → 0.834 (3.4x)** | ROC-AUC hides that entirely (0.931 vs 0.965 reads as "almost as good"). The **hybrid was slightly worse** than plain XGBoost (0.829 vs 0.834) and is reported as the negative result it is |

### Datasets

| Folder | Data | Size | Fraud rate |
|---|---|---|---|
| 01 | Real — Credit Card Fraud 2023 (Kaggle) | 568,629 transactions | balanced by construction |
| 02 | Synthetic — Chilean e-commerce / Redcompra-style | generated per run | configurable |
| 03 | Synthetic — Chilean interbank transfer network | 43,009 transfers | unlabeled (UAF typologies as ground truth) |
| 04 | Real — ULB/Worldline European card transactions | 284,807 transactions | 492 frauds (**0.172%**) |

---

## Evidence

### How much you lose without labels

![Precision-Recall: unsupervised vs supervised vs hybrid](04-autoencoder-vs-supervised/outputs/figures/precision_recall_curves.png)

**How to read it.** Five models on the identical held-out test set of the real ULB dataset. The dashed line at the bottom is chance (0.0017 prevalence). The three unsupervised models saw **zero fraud labels** during training — the realistic day-one scenario.

The vertical distance between the blue curve (plain autoencoder) and the green one (supervised XGBoost) is the cost of not yet having confirmed labels, and it is large: PR-AUC 0.242 against 0.834. Note also how far apart the three unsupervised models are from each other: **Deep SVDD (teal) stays within ~0.1 precision of supervised XGBoost across most of the range**, the VAE (orange) sits well below it, and the plain autoencoder (blue) is worst by a wide margin. The choice of unsupervised objective matters more than the fact of being unsupervised.

### Why the top of the alert queue is not the best place to stop

![Precision/recall by alert budget](03-graph-based-aml-detection/outputs/figures/precision_recall_sweep.png)

**How to read it.** The x axis is the fraction of accounts flagged for review — the analyst team's capacity. Red is precision on those alerts, blue is recall against the injected UAF typologies.

The intuition says precision should be highest at the very top of the ranking and fall from there. **It doesn't.** Precision climbs from 0.400 at a 1% budget to a peak of **0.517 at 3%**, and only then declines. The handful of highest-scored accounts are extreme outliers that are not all money laundering; widening the budget slightly brings in genuine typologies before noise takes over. A team that only ever reviewed its top 1% would be operating on the wrong side of that peak.

### Cost-calibrated decisions, not a 0.5 cutoff

![Model comparison](01-credit-card-fraud-multilang/outputs/reports/model_comparison.png)

**How to read it — including a trap in the right-hand panel.** On the left, PR-AUC for the three main models on a zoomed axis (0.990–1.000): they are effectively tied, which makes the ranking metric useless for choosing between them. On the right, the percentage by which threshold calibration cut the business cost relative to each model's *own* 0.5-threshold baseline.

Read naively, the right panel says **LogReg + SMOTE is the best model** — a 47.9% cost reduction, ahead of XGBoost's 44.7%. That reading is wrong, and the reason is worth internalizing: the percentage is relative to each model's own starting point. In absolute terms LogReg's calibrated cost is **134,440** against XGBoost's **260** — a factor of 517. A weak model with a terrible baseline can post an excellent-looking "improvement".

The absolute table lives in [the folder's README](01-credit-card-fraud-multilang): 134,440 for LogReg + SMOTE, 1,150 for the MLP, 770 for CatBoost, 260 for XGBoost, and **180** for the Optuna-tuned XGBoost.

---

## The pattern across all four

The four techniques were built independently, on different data, for different operating contexts. They converge on the same lesson:

> **The metric you optimize is not the metric that decides.**

- In **01**, PR-AUC saturates at 1.0000 and stops discriminating between models — while the cost-calibrated threshold keeps separating them by a factor of 700x.
- In **04**, ROC-AUC says the unsupervised autoencoder is "almost as good" as supervised XGBoost (0.931 vs 0.965). PR-AUC says it recovers less than a third as much (0.242 vs 0.834). Both are correct; only one is useful under a 0.172% fraud rate.
- In **03**, the ranking is good (ROC-AUC 0.893) but the operating point that maximizes precision isn't where anyone would look for it.
- In **02**, the deployed threshold minimizes an explicit CLP cost function rather than defaulting to 0.5 — which is what produces 0 false alarms at 0.955 recall.

A second thread worth naming: **the negative results are reported as findings.** The hybrid in 04 was slightly worse than plain XGBoost and stayed in. The MLP in 01 lands in the same metric tier as the gradient-boosted trees but at a higher business cost, and stayed in. In 03, point-anomaly typologies rank only in the top 15.6–22.5% while structural ones rank in the top 2.5–3.5%, because account-level aggregation dilutes single-transaction outliers — stated plainly rather than smoothed over.

---

## Why one repo instead of four

Each technique is real, runnable and independently tested — this isn't about hiding scope, it's about representing it accurately. Four repos with overlapping "fraud detection" descriptions read as repetition; one lab with four clearly differentiated techniques (multi-language systems work, real-time serving, graph/unsupervised AML, and a direct supervised-vs-unsupervised comparison) reads as what it actually is: a systematic study of the same problem from different angles.

Related work on this profile: [`Proyectos_ML_anomalias`](https://github.com/Rxyxs/Proyectos_ML_anomalias) takes the unsupervised angle much further — 16 detector families benchmarked on the same split, conformal coverage guarantees, and an operational layer that turns scores into thresholds and money. Its Deep SVDD result independently reproduces what folder 04 finds here.

## Running a technique

Each folder is self-contained:

```bash
cd 0N-technique-name
python -m venv venv
venv/Scripts/pip install -r requirements.txt   # Windows
python <entry_point>.py
```

See the folder's own README for the exact entry point, the full results table from an actual run, the architecture diagram, and the honest negative findings.

## Author

Pablo Reyes — [github.com/Rxyxs](https://github.com/Rxyxs)
Code: MIT — see [LICENSE](LICENSE)
