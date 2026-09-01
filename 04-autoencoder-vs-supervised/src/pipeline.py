"""Pipeline end-to-end: descarga (si falta) -> autoencoder no supervisado
-> XGBoost supervisado -> experimento hibrido -> comparacion -> figuras.

Uso:
    python -m src.pipeline
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.download_dataset import main as descargar_datos
from src.evaluation.compare_models import main as comparar_modelos
from src.evaluation.optimize_thresholds import main as optimizar_umbrales
from src.models.train_autoencoder import main as entrenar_autoencoder
from src.models.train_deep_svdd import main as entrenar_deep_svdd
from src.models.train_hybrid import main as entrenar_hibrido
from src.models.train_supervised import main as entrenar_supervisado
from src.models.train_vae import main as entrenar_vae
from src.visualization.plots import main as generar_figuras


def main():
    print("=" * 70)
    print("1/7 - Verificando / descargando dataset real (OpenML, ULB)")
    print("=" * 70)
    descargar_datos()

    print("\n" + "=" * 70)
    print("2/7 - Entrenando autoencoder no supervisado (solo transacciones normales)")
    print("=" * 70)
    entrenar_autoencoder()

    print("\n" + "=" * 70)
    print("3/7 - Entrenando VAE y Deep SVDD (arquitecturas alternativas no supervisadas)")
    print("=" * 70)
    entrenar_vae()
    entrenar_deep_svdd()

    print("\n" + "=" * 70)
    print("4/7 - Entrenando XGBoost supervisado (con etiquetas de fraude)")
    print("=" * 70)
    entrenar_supervisado()

    print("\n" + "=" * 70)
    print("5/7 - Experimento hibrido (XGBoost + feature del autoencoder)")
    print("=" * 70)
    entrenar_hibrido()

    print("\n" + "=" * 70)
    print("6/7 - Comparando modelos y optimizando umbrales sensibles a costo")
    print("=" * 70)
    comparar_modelos()
    optimizar_umbrales()

    print("\n" + "=" * 70)
    print("7/7 - Generando figuras")
    print("=" * 70)
    generar_figuras()

    print("\nPipeline completo. Ver outputs/reports/comparison_report.md y outputs/figures/")


if __name__ == "__main__":
    main()
