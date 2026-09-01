"""Orquestador end-to-end: generacion de datos sinteticos -> ingenieria de
atributos -> entrenamiento (Autoencoder + CatBoost/XGBoost costo-sensible).

Uso:
    python run_pipeline.py
"""
from __future__ import annotations

from pathlib import Path

from src.data.generate_transactions import generate_dataset
from src.models import train as train_module

BASE_DIR = Path(__file__).resolve().parent


def main():
    raw_dir = BASE_DIR / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("[1/2] Generando 100,000 transacciones sinteticas (Chile, < 1% fraude)...")
    raw = generate_dataset()
    raw_path = raw_dir / "transactions.parquet"
    raw.to_parquet(raw_path, index=False)
    fraud_pct = 100 * raw["is_fraud"].mean()
    print(f"       {len(raw):,} transacciones -> {raw_path} (fraude: {fraud_pct:.3f}%)")

    print("[2/2] Ingenieria de atributos + Autoencoder + CatBoost/XGBoost costo-sensible...")
    train_module.main()

    print(
        "\nPipeline completo. Para servir el modelo:\n"
        "  API:       uvicorn src.api.main:app --reload\n"
        "  Dashboard: streamlit run src/dashboard/app.py\n"
        "  Tests:     pytest -v"
    )


if __name__ == "__main__":
    main()
