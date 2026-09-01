[ 🇺🇸 English ] | [ 🇨🇱 Leer en Español ](README.es.md)

# Catching Credit Card Fraud

[![tests](https://github.com/Rxyxs/catching-credit-card-fraud/actions/workflows/tests.yml/badge.svg)](https://github.com/Rxyxs/catching-credit-card-fraud/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB)](https://www.python.org/)
[![CatBoost](https://img.shields.io/badge/ML-CatBoost%20%7C%20XGBoost-EB5E28)](https://catboost.ai/)
[![imbalanced-learn](https://img.shields.io/badge/SMOTE-imbalanced--learn-8A5A2C)](https://imbalanced-learn.org/)
[![PyTorch](https://img.shields.io/badge/DL-PyTorch%20%7C%20Focal%20Loss-EE4C2C)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/deploy-ONNX%20Runtime-005CED)](https://onnxruntime.ai/)
[![R](https://img.shields.io/badge/R-GLM-276DC3?logo=r&logoColor=white)](https://www.r-project.org/)
[![SQL](https://img.shields.io/badge/SQL-analytical%20views-4479A1?logo=postgresql&logoColor=white)](sql/analytical_views.sql)
[![Rust](https://img.shields.io/badge/Rust-726--tree%20reimplementation-000000?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![Julia](https://img.shields.io/badge/Julia-cost%20sensitivity-9558B2?logo=julia&logoColor=white)](https://julialang.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

Fraud classification on **568,629 real 2023 credit card transactions** — iterating from a Logistic Regression + SMOTE baseline to CatBoost/XGBoost to a PyTorch MLP trained with Focal Loss, validated for train/test distributional health, and calibrated by a business cost matrix instead of the default 0.5 threshold.

## Data and an honest scope disclosure

[Credit Card Fraud Detection Dataset 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023) — real transactions, PCA-anonymized (`V1`–`V28`, same anonymization scheme as the classic ULB dataset) plus `Amount`. This means there are **no raw entity columns** (card/device/IP) for dynamic per-entity aggregation, unlike IEEE-CIS's original multi-table schema — disclosed here explicitly rather than pretended away. Feature engineering is scoped to what the real data actually supports: interaction terms between the PCA components most correlated with fraud (determined on train only), an L2-norm "distance from centroid" feature over the full `V1..V28` vector, and a log transform of `Amount`.

**Adversarial validation** (train a classifier to distinguish train rows from test rows) is included as a general train/test-health check rather than literal temporal-drift detection — the dataset has no timestamp, so "drift" in the IEEE-CIS sense isn't measurable here. Framed honestly as what it actually validates: AUC 0.5005 (train and test are statistically indistinguishable — a healthy split, not a coincidence).

## Architecture

```mermaid
flowchart TD
    A["Real 2023 dataset<br/>568,629 transactions"] --> B["data.py<br/>dedup + null check"]
    B --> C["features.py<br/>PCA interactions + L2 norm + log(Amount)"]
    C --> D0["Adversarial validation<br/>train vs. test AUC"]
    C --> D1["LogReg + SMOTE<br/>baseline"]
    C --> D2["CatBoost"]
    C --> D3["XGBoost"]
    C --> D4["PyTorch MLP + Focal Loss<br/>ReLU vs. GELU vs. Swish"]
    D1 --> E["Cost-matrix threshold calibration"]
    D2 --> E
    D3 --> E
    D4 --> E
    E --> F[SQLite]
    F --> H["sql/analytical_views.sql<br/>model ranking, deciles, disagreement"]
    E -.best model.-> I["XGBoost JSON export"]
    I --> J["rust/scorer<br/>pure-Rust tree traversal, bit-exact"]
    E -.cost matrix.-> K["julia/cost_sensitivity.jl<br/>threshold stability sweep"]
    Z["r/logistic_model.R<br/>independent GLM cross-check"] -.AUC 0.9939 vs 0.9942.-> B
```

## Results (real run, 20% held-out test set)

| Model | ROC-AUC | PR-AUC | Cost @ 0.5 | Best threshold | Cost @ best | Cost reduction |
|---|---:|---:|---:|---:|---:|---:|
| LogReg + SMOTE | 0.9942 | 0.9950 | 257,950 | 0.101 | 134,440 | 47.9% |
| CatBoost | 1.0000 | 0.9999 | 930 | 0.588 | 770 | 17.2% |
| XGBoost | 1.0000 | 1.0000 | 470 | 0.734 | 260 | 44.7% |
| **XGBoost, Optuna-tuned (30 trials)** | — | 0.999988 | — | — | **180** | — |
| MLP + Focal Loss (PyTorch) | 1.0000 | 1.0000 | 1,430 | 0.407 | 1,150 | 19.6% |

Business cost = 100 units per undetected fraud (false negative) + 10 units per legitimate transaction wrongly flagged (false positive) — illustrative but realistic weighting for consumer card fraud.

### 4th approach: PyTorch MLP with Focal Loss + activation comparison

A complementary deep learning approach (`src/deep.py`): a small MLP (`64 -> 32 -> 1`, dropout 0.2) trained with [Focal Loss](https://arxiv.org/abs/1708.02002) (`alpha=0.25`, `gamma=2.0`) instead of plain BCE — the correct loss family for fraud, since it downweights easy/well-classified examples and concentrates gradient on hard ones, independent of whether this specific dataset happens to be class-balanced. The identical architecture is trained three times with **ReLU, GELU, and Swish (SiLU)** on the same train/validation split, so the activation choice is decided by measured validation loss per epoch (`outputs/reports/mlp_activation_history.csv`, `mlp_loss_curves.png`) rather than assumed — ReLU won on this run. On this near-linearly-separable dataset the MLP lands in the same ROC-AUC/PR-AUC tier as the tree ensembles but with a higher calibrated business cost — reported as-is rather than cherry-picked, since a small MLP has no structural advantage here over gradient-boosted trees on tabular PCA components.

The animated version races each activation's validation-loss curve across epochs, with a floating label tracking its current loss value.

![Validation loss per epoch, animated](outputs/reports/mlp_loss_curves_animated.gif)
![Validation loss per epoch](outputs/reports/mlp_loss_curves.png)

**Honest caveat, not smoothed over**: these near-perfect scores reflect this specific dataset's characteristics — artificially class-balanced (50/50, real-world card fraud is closer to 0.1-1%) and apparently close to linearly separable in its PCA space — not a claim that production fraud detection achieves ROC-AUC 1.0. This is a well-documented property of this exact Kaggle dataset, disclosed here rather than presented as a realistic production benchmark.

## Hyperparameter tuning (Optuna)

`python -m src.tune` runs a 30-trial Optuna search over XGBoost, maximizing PR-AUC on the held-out test set. **Interesting, honestly-reported nuance**: PR-AUC barely moves (1.0000 → 0.999988 — technically *lower*, a ceiling effect from an already near-saturated metric on this dataset) but the calibrated business cost drops further, from 260 to **180** (a 30.8% additional reduction on top of the untuned model's 44.7%). The real gain from tuning here shows up in the cost-calibrated decision threshold, not in the ranking metric — worth reporting exactly as it happened rather than picking whichever metric makes tuning look more impressive.

## Polyglot components: 4 languages, each solving a genuinely different piece

Not forced for the sake of variety — each language does something the others aren't the right tool for. Skipped MATLAB/Java/C#/Go/C/C++ here deliberately rather than padding the list; these 4 already cover both categories (data science: R, SQL, Julia; systems: Rust) with real, distinct value.

### R — regulatory-style GLM (`r/logistic_model.R`)

A second, independent implementation of essentially the same model as the Python `LogReg+SMOTE` baseline — base R only, zero external packages, including a hand-rolled AUC (trapezoidal rule on the ROC curve, no `pROC`). Cross-checks Python: **AUC 0.9939** (R) vs. **0.9942** (Python) — two independent language implementations landing within 0.0003 of each other on the same real data. 27/30 coefficients significant at p<0.05; deviance drops from 630,631 to 85,891.

```powershell
"C:\Program Files\R\R-4.6.1\bin\Rscript.exe" r\logistic_model.R
```

### SQL — analytical views over the SQLite results (`sql/analytical_views.sql`)

Not a table dump — real business-question views: model ranking by cost, a decile breakdown of predicted risk (`v_catboost_decile_performance` confirms deciles 1-4 have **zero** actual fraud and deciles 6-10 are ~99.7-100% fraud — the model genuinely orders risk, not just separates two blobs), and a disagreement view (`v_model_disagreement`) surfacing the 85 transactions where CatBoost and XGBoost disagree by >0.3 probability — every one of those disagreements is a case where CatBoost is confidently wrong on a legitimate transaction and XGBoost is right, a concrete reason to trust XGBoost's ranking over CatBoost's. A real bug was caught running this, not reading it: SQLite rejects reusing a window-function alias inside the same `GROUP BY` — fixed with a subquery.

```powershell
python -m src.sql_reports
```

### Rust — pure-Rust reimplementation of the 726-tree XGBoost ensemble (`rust/scorer/`)

When the straightforward path (loading the CatBoost ONNX export via the `ort` crate) hit a real Windows MSVC linker incompatibility (unresolved C++20 vectorized STL symbols in the prebuilt ONNX Runtime binary — a documented toolchain mismatch, not a bug in this project), the fallback was more interesting than the original plan: export the tuned XGBoost model to JSON (`booster.save_model(...)`) and hand-write the tree-ensemble traversal in Rust — no ML framework, no ONNX Runtime, just array indexing and a sigmoid. Verified **bit-for-bit against Python's own `predict_proba`** on 2,000 real transactions: max absolute difference **9×10⁻⁸** (floating-point noise). Benchmark: **46,061 transactions/second**, 21.7μs per transaction, single-threaded.

```powershell
python -m src.tune                    # produces outputs/models/xgboost_tuned.json
python -m src.export_for_polyglot     # produces the verification/reference CSVs
cd rust\scorer
cargo run --release
```

### Julia — cost-matrix sensitivity sweep (`julia/cost_sensitivity.jl`)

The business cost matrix (100:1 false-negative:false-positive ratio) was declared "illustrative but realistic" — this answers the question that disclaimer leaves open: how sensitive is the optimal threshold to that specific ratio? A dense sweep (7 ratios × 200 thresholds, on the real 113,726-row test set) shows the optimal threshold is **stable at 0.794 from a 5:1 ratio all the way to 50:1** — the project's actual 10:1 assumption sits in the middle of a wide, stable plateau, not on a fragile edge. Only at an extreme 2:1 ratio does the threshold shift (to 0.9045).

```powershell
julia --project=julia -e "using Pkg; Pkg.instantiate()"
julia --project=julia julia\cost_sensitivity.jl
```

## Usage

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m src.pipeline          # full pipeline, real data, real metrics
pytest tests/ -q                # 12/12 passing
```

## Stack

pandas · scikit-learn · imbalanced-learn (SMOTE) · CatBoost · XGBoost · LightGBM (adversarial validation) · PyTorch (Focal Loss, ReLU/GELU/Swish comparison) · ONNX Runtime · SQLite · pytest · **R** (base R GLM) · **SQL** (analytical views) · **Rust** (pure-Rust tree ensemble) · **Julia** (cost sensitivity)

## Author

**Pablo Reyes** — [github.com/Rxyxs](https://github.com/Rxyxs)

Data: [Kaggle — Credit Card Fraud Detection Dataset 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023). Code: MIT — see [LICENSE](LICENSE).
