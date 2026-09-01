[ 🇺🇸 Read in English ](README.md) | [ 🇨🇱 Español ]

# Fraud Detection Techniques Lab

Cuatro enfoques de detección de fraude/AML en un solo laboratorio, cada uno aislando una técnica distinta sobre un perfil de datos distinto — fraude real con tarjetas, e-commerce en tiempo real, tipologías AML interbancarias y una comparación directa autoencoder vs. supervisado. Cada carpeta es autocontenida, con su propio README, dependencias y tests. Este repo reemplaza cuatro repos separados de una sola técnica que antes vivían en este perfil.

## Técnicas

| # | Técnica | Carpeta | Qué hace |
|---|---|---|---|
| 01 | Pipeline multi-lenguaje sobre datos reales | [`01-credit-card-fraud-multilang`](01-credit-card-fraud-multilang) | Dataset real de fraude con tarjetas 2023 (568k transacciones): Python (LogisticRegression+SMOTE, CatBoost, XGBoost con validación adversaria), R, Julia, Rust y SQL, cada uno a cargo de una etapa distinta del mismo pipeline, más exportación ONNX. |
| 02 | Detección en tiempo real | [`02-realtime-ecommerce-fraud`](02-realtime-ecommerce-fraud) | Scoring de fraude de baja latencia con FastAPI para e-commerce/pagos con tarjeta en Chile: pre-filtro autoencoder PyTorch alimentando CatBoost/XGBoost sensible al costo. |
| 03 | Detección no supervisada basada en grafos | [`03-graph-based-aml-detection`](03-graph-based-aml-detection) | Tipologías AML sobre una red sintética de transferencias interbancarias chilenas: features de grafo con NetworkX + ensamble no supervisado PyOD (IForest+COPOD+ECOD). |
| 04 | Autoencoder vs. supervisado, cara a cara | [`04-autoencoder-vs-supervised`](04-autoencoder-vs-supervised) | El mismo dataset real ULB/Worldline evaluado de dos formas — autoencoder PyTorch no supervisado vs. XGBoost supervisado vs. híbrido — para cuantificar la brecha entre un sistema recién iniciado y uno maduro. |

## Por qué un repo en vez de cuatro

Cada técnica es real, ejecutable y probada de forma independiente — esto no es esconder alcance, es representarlo con precisión. Cuatro repos con descripciones de "detección de fraude" superpuestas se leen como repetición; un laboratorio con cuatro técnicas claramente diferenciadas (trabajo de sistemas multi-lenguaje, serving en tiempo real, AML no supervisado basado en grafos, y una comparación directa supervisado-vs-no-supervisado) se lee como lo que realmente es: un estudio sistemático del mismo problema desde distintos ángulos.

## Cómo correr una técnica

Cada carpeta es autocontenida:

```bash
cd 0N-nombre-tecnica
python -m venv venv
venv/Scripts/pip install -r requirements.txt   # Windows
python <entry_point>.py
```

Ver el README de cada carpeta para el entry point exacto, resultados reales de una corrida real, y cualquier hallazgo negativo honesto.

## Autor

Pablo Reyes — [github.com/Rxyxs](https://github.com/Rxyxs)
Código: MIT — ver [LICENSE](LICENSE)
