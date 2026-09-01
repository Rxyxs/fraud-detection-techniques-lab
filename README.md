[ 🇺🇸 English ] | [ 🇨🇱 Leer en Español ](README.es.md)

# Fraud Detection Techniques Lab

Four fraud/AML detection approaches in one lab, each isolating a different technique on a different data profile — real-world card fraud, real-time e-commerce, interbank AML typologies, and a direct autoencoder-vs-supervised comparison. Each folder is self-contained with its own README, dependencies, and tests. This repo replaces four separate single-technique repos that used to live on this profile.

## Techniques

| # | Technique | Folder | What it does |
|---|---|---|---|
| 01 | Multi-language pipeline on real data | [`01-credit-card-fraud-multilang`](01-credit-card-fraud-multilang) | Real 2023 card-fraud dataset (568k transactions): Python (LogisticRegression+SMOTE, CatBoost, XGBoost with adversarial validation), R, Julia, Rust, and SQL each handling a different stage of the same pipeline, plus ONNX export. |
| 02 | Real-time detection | [`02-realtime-ecommerce-fraud`](02-realtime-ecommerce-fraud) | Low-latency FastAPI fraud scoring for Chilean e-commerce/card payments: PyTorch autoencoder pre-filter feeding cost-sensitive CatBoost/XGBoost. |
| 03 | Graph-based unsupervised detection | [`03-graph-based-aml-detection`](03-graph-based-aml-detection) | AML typologies over a synthetic Chilean interbank transfer network: NetworkX graph features + PyOD unsupervised ensemble (IForest+COPOD+ECOD). |
| 04 | Autoencoder vs. supervised, head-to-head | [`04-autoencoder-vs-supervised`](04-autoencoder-vs-supervised) | Same real ULB/Worldline fraud dataset scored two ways — unsupervised PyTorch autoencoder vs. supervised XGBoost vs. a hybrid — to quantify the cold-start-vs-mature-system gap. |

## Why one repo instead of four

Each technique is real, runnable, and independently tested — this isn't about hiding scope, it's about representing it accurately. Four repos with overlapping "fraud detection" descriptions read as repetition; one lab with four clearly differentiated techniques (multi-language systems work, real-time serving, graph/unsupervised AML, and a direct supervised-vs-unsupervised comparison) reads as what it actually is: a systematic study of the same problem from different angles.

## Running a technique

Each folder is self-contained:

```bash
cd 0N-technique-name
python -m venv venv
venv/Scripts/pip install -r requirements.txt   # Windows
python <entry_point>.py
```

See the folder's own README for the exact entry point, real results from an actual run, and any honest negative findings.

## Author

Pablo Reyes — [github.com/Rxyxs](https://github.com/Rxyxs)
Code: MIT — see [LICENSE](LICENSE)
