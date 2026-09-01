"""Descarga el dataset real de fraude con tarjetas de credito (ULB/Worldline,
via OpenML, dataset id 1597) usado por este proyecto.

Este es el dataset publico mas conocido de deteccion de fraude: 284,807
transacciones de tarjetahabientes europeos en septiembre de 2013, con 492
fraudes confirmados (0.172%). Las columnas V1-V28 son componentes PCA de los
features originales (no divulgados por confidencialidad); `Time` y `Amount`
no fueron transformados. `Class` es la etiqueta real (1 = fraude
confirmado, 0 = transaccion legitima).

Fuente: OpenML (https://www.openml.org/d/1597), espejo del dataset original
publicado por Pozzolo et al. (ULB Machine Learning Group) en Kaggle
(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

Uso:
    python data/download_dataset.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import requests

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

OPENML_CSV_URL = "https://www.openml.org/data/get_csv/1673544/phpKo8OWT"
RAW_DIR = Path(__file__).parent / "raw"
OUTPUT_PATH = RAW_DIR / "creditcard.csv"
EXPECTED_ROWS = 284_807


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "rb") as f:
            n_lines = sum(1 for _ in f) - 1
        if n_lines == EXPECTED_ROWS:
            print(f"Dataset ya presente y completo en {OUTPUT_PATH} ({n_lines:,} filas). Nada que hacer.")
            return

    print(f"Descargando dataset real de fraude (~150 MB) desde OpenML...\n  {OPENML_CSV_URL}")
    response = requests.get(OPENML_CSV_URL, timeout=600, stream=True)
    response.raise_for_status()

    hasher = hashlib.md5()
    with open(OUTPUT_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            hasher.update(chunk)

    with open(OUTPUT_PATH, "rb") as f:
        n_lines = sum(1 for _ in f) - 1

    print(f"Descarga completa: {OUTPUT_PATH} ({n_lines:,} filas, md5={hasher.hexdigest()})")
    if n_lines != EXPECTED_ROWS:
        print(f"ADVERTENCIA: se esperaban {EXPECTED_ROWS:,} filas, se obtuvieron {n_lines:,}.")


if __name__ == "__main__":
    main()
