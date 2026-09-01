"""Generador sintetico de transferencias interbancarias TEF (Chile).

Crea una red de cuentas y transferencias con cuatro tipologias de Lavado de
Activos inyectadas de forma controlada (con etiqueta oculta al motor no
supervisado, usada solo para evaluar el pipeline despues de la deteccion):

    1. pitufeo            - structuring: fraccionamiento de un monto grande
                             en muchas transferencias bajo un umbral, desde
                             varias cuentas "mula" hacia una o dos cuentas
                             colectoras, en una ventana corta de tiempo.
    2. cuenta_puente      - layering: una cadena de cuentas que recibe y
                             reenvia fondos casi de inmediato (bajo tiempo de
                             retencion, monto de entrada ~= monto de salida).
    3. rafaga_cuenta_nueva- fan-in: una cuenta recien abierta recibe una
                             rafaga de transferencias de muchos remitentes
                             distintos en pocas horas y luego las reenvia.
    4. monto_inusual      - outlier puntual: una cuenta con actividad
                             habitualmente baja realiza una transferencia muy
                             por sobre su propio historial.

AVISO IMPORTANTE: todos los datos generados aqui son 100% sinteticos. Los
nombres de bancos chilenos se usan solo para dar contexto de dominio
realista; ninguna cifra, cuenta o patron representa actividad real de una
institucion o persona. El umbral UMBRAL_ESTRUCTURACION_CLP es un parametro
ilustrativo para demostrar la tipologia de fraccionamiento y no corresponde
a un monto oficial publicado por la UAF.
"""

from __future__ import annotations

import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

SEED = 42
N_ACCOUNTS = 2000
SIMULATION_DAYS = 90
END_DATE = datetime(2026, 8, 24)
START_DATE = END_DATE - timedelta(days=SIMULATION_DAYS)

# Parametro ilustrativo: umbral interno de fraccionamiento que un actor
# ilicito buscaria evadir. No es un valor legal oficial de la UAF.
UMBRAL_ESTRUCTURACION_CLP = 5_000_000

N_NORMAL_TRANSFERS = 42_000
N_PITUFEO_CASES = 18
N_PUENTE_CASES = 18
N_RAFAGA_CASES = 18
N_INUSUAL_CASES = 22

BANCOS = [
    "Banco de Chile",
    "Banco Santander-Chile",
    "BancoEstado",
    "Banco de Credito e Inversiones (BCI)",
    "Scotiabank Chile",
    "Banco Itau Chile",
    "Banco Security",
    "Banco Falabella",
    "Banco BICE",
    "Banco Consorcio",
]

REGIONES = [
    "Region Metropolitana", "Valparaiso", "Biobio", "Antofagasta",
    "Maule", "Araucania", "Coquimbo", "Los Lagos", "O'Higgins", "Atacama",
]
REGION_WEIGHTS = np.array([0.42, 0.12, 0.09, 0.07, 0.06, 0.06, 0.05, 0.05, 0.04, 0.04])

TIPO_CLIENTE = ["Persona Natural", "Persona Juridica"]
TIPO_CLIENTE_WEIGHTS = np.array([0.82, 0.18])

CANALES = ["App Movil", "Banca en Linea", "Sucursal", "API Empresas"]
CANAL_WEIGHTS = np.array([0.55, 0.32, 0.06, 0.07])

OUT_DIR = Path(__file__).parent / "synthetic"


def _sample_timestamp(rng: np.random.Generator, day_lo: int, day_hi: int) -> datetime:
    """Fecha uniforme dentro del rango de dias, hora con sesgo a horario habil."""
    day_offset = int(rng.integers(day_lo, day_hi))
    if rng.random() < 0.93:
        hour = int(np.clip(rng.normal(14, 3.5), 0, 23))
    else:
        hour = int(rng.integers(22, 24)) if rng.random() < 0.5 else int(rng.integers(0, 6))
    minute = int(rng.integers(0, 60))
    second = int(rng.integers(0, 60))
    return START_DATE + timedelta(days=day_offset, hours=hour, minutes=minute, seconds=second)


@dataclass
class TransferBatch:
    rows: list[dict] = field(default_factory=list)

    def add(self, **kwargs):
        self.rows.append(kwargs)


def generate_accounts(rng: np.random.Generator, n: int) -> pl.DataFrame:
    tipo = rng.choice(TIPO_CLIENTE, size=n, p=TIPO_CLIENTE_WEIGHTS)
    banco = rng.choice(BANCOS, size=n)
    region = rng.choice(REGIONES, size=n, p=REGION_WEIGHTS)
    # cuentas normales: antiguedad entre 60 dias y 15 anios antes del cierre
    antiguedad_dias = rng.integers(60, 15 * 365, size=n)
    fecha_apertura = [END_DATE - timedelta(days=int(d)) for d in antiguedad_dias]
    riesgo = rng.choice(["Bajo", "Medio", "Alto"], size=n, p=[0.75, 0.20, 0.05])
    account_id = [f"CTA{idx:05d}" for idx in range(n)]
    # popularidad para conexion preferencial: la mayoria de cuentas es de
    # baja actividad, un pequenio grupo actua como hub (comercios, nomina)
    popularidad = rng.pareto(a=2.0, size=n) + 0.1

    return pl.DataFrame(
        {
            "account_id": account_id,
            "tipo_cliente": tipo,
            "banco": banco,
            "region": region,
            "fecha_apertura": fecha_apertura,
            "segmento_riesgo_declarado": riesgo,
            "popularidad": popularidad,
        }
    )


def generate_normal_transfers(rng: np.random.Generator, accounts: pl.DataFrame, n: int) -> TransferBatch:
    batch = TransferBatch()
    ids = accounts["account_id"].to_list()
    tipo_map = dict(zip(ids, accounts["tipo_cliente"].to_list()))
    banco_map = dict(zip(ids, accounts["banco"].to_list()))
    pop = accounts["popularidad"].to_numpy()
    weights = pop / pop.sum()
    n_accounts = len(ids)

    origenes_idx = rng.choice(n_accounts, size=n, p=weights)
    destinos_idx = rng.choice(n_accounts, size=n, p=weights)

    for i in range(n):
        o_idx, d_idx = int(origenes_idx[i]), int(destinos_idx[i])
        if o_idx == d_idx:
            d_idx = (d_idx + 1) % n_accounts
        origen, destino = ids[o_idx], ids[d_idx]
        es_empresa = tipo_map[origen] == "Persona Juridica"
        base = rng.lognormal(mean=13.3 if es_empresa else 11.8, sigma=1.0)
        monto = float(np.clip(base, 2_000, 90_000_000))
        ts = _sample_timestamp(rng, 0, SIMULATION_DAYS)
        batch.add(
            origen=origen, destino=destino, monto_clp=round(monto, 0),
            timestamp=ts, banco_origen=banco_map[origen], banco_destino=banco_map[destino],
            canal=rng.choice(CANALES, p=CANAL_WEIGHTS), tipologia="normal", es_ilicito=False,
            caso_id=None,
        )
    return batch


def inject_pitufeo(rng: np.random.Generator, ids: list[str], banco_map: dict, case_id: str) -> tuple[TransferBatch, set[str]]:
    batch = TransferBatch()
    n_mulas = int(rng.integers(2, 5))
    mulas = list(rng.choice(ids, size=n_mulas, replace=False))
    n_colectoras = int(rng.integers(1, 3))
    colectoras = [a for a in rng.choice(ids, size=n_colectoras, replace=False) if a not in mulas]
    if not colectoras:
        colectoras = [rng.choice([a for a in ids if a not in mulas])]

    ventana_dias = int(rng.integers(2, 10))
    dia_inicio = int(rng.integers(0, SIMULATION_DAYS - ventana_dias - 1))
    n_transfers = int(rng.integers(10, 30))

    etiquetadas = set(mulas) | set(colectoras)
    for _ in range(n_transfers):
        origen = str(rng.choice(mulas))
        destino = str(rng.choice(colectoras))
        fraccion = rng.uniform(0.55, 0.985)
        monto = round(UMBRAL_ESTRUCTURACION_CLP * fraccion, 0)
        ts = _sample_timestamp(rng, dia_inicio, dia_inicio + ventana_dias)
        batch.add(
            origen=origen, destino=destino, monto_clp=monto, timestamp=ts,
            banco_origen=banco_map[origen], banco_destino=banco_map[destino],
            canal=rng.choice(CANALES, p=CANAL_WEIGHTS), tipologia="pitufeo", es_ilicito=True,
            caso_id=case_id,
        )
    return batch, etiquetadas


def inject_cuenta_puente(rng: np.random.Generator, ids: list[str], banco_map: dict, case_id: str) -> tuple[TransferBatch, set[str]]:
    batch = TransferBatch()
    k = int(rng.integers(3, 7))
    cadena = list(rng.choice(ids, size=k, replace=False))
    monto = float(rng.uniform(8_000_000, 60_000_000))
    dia_inicio = int(rng.integers(0, SIMULATION_DAYS - 2))
    t0 = _sample_timestamp(rng, dia_inicio, dia_inicio + 1)

    puentes = set(cadena[1:-1])
    ts = t0
    for i in range(k - 1):
        origen, destino = cadena[i], cadena[i + 1]
        skim = rng.uniform(0.01, 0.03)
        monto *= (1 - skim)
        ts = ts + timedelta(minutes=int(rng.integers(15, 240)))
        batch.add(
            origen=origen, destino=destino, monto_clp=round(monto, 0), timestamp=ts,
            banco_origen=banco_map[origen], banco_destino=banco_map[destino],
            canal=rng.choice(CANALES, p=CANAL_WEIGHTS), tipologia="cuenta_puente", es_ilicito=True,
            caso_id=case_id,
        )
    return batch, puentes


def inject_rafaga_cuenta_nueva(rng: np.random.Generator, ids: list[str], banco_map: dict, case_id: str) -> tuple[TransferBatch, set[str], str, datetime]:
    batch = TransferBatch()
    objetivo = str(rng.choice(ids))
    n_remitentes = int(rng.integers(15, 40))
    remitentes = list(rng.choice([a for a in ids if a != objetivo], size=n_remitentes, replace=False))

    ventana_horas = int(rng.integers(12, 72))
    dia_inicio = int(rng.integers(3, SIMULATION_DAYS - 3))
    t0 = _sample_timestamp(rng, dia_inicio, dia_inicio + 1)
    nueva_apertura = t0 - timedelta(days=int(rng.integers(3, 15)))

    total_recibido = 0.0
    for remitente in remitentes:
        monto = float(rng.uniform(100_000, 2_000_000))
        total_recibido += monto
        ts = t0 + timedelta(hours=float(rng.uniform(0, ventana_horas)))
        batch.add(
            origen=remitente, destino=objetivo, monto_clp=round(monto, 0), timestamp=ts,
            banco_origen=banco_map[remitente], banco_destino=banco_map[objetivo],
            canal=rng.choice(CANALES, p=CANAL_WEIGHTS), tipologia="rafaga_cuenta_nueva", es_ilicito=True,
            caso_id=case_id,
        )

    n_reenvios = int(rng.integers(1, 4))
    destinos_reenvio = list(rng.choice([a for a in ids if a != objetivo], size=n_reenvios, replace=False))
    monto_reenvio_total = total_recibido * rng.uniform(0.80, 0.96)
    for destino in destinos_reenvio:
        monto = round(monto_reenvio_total / n_reenvios, 0)
        ts = t0 + timedelta(hours=float(rng.uniform(ventana_horas, ventana_horas + 12)))
        batch.add(
            origen=objetivo, destino=destino, monto_clp=monto, timestamp=ts,
            banco_origen=banco_map[objetivo], banco_destino=banco_map[destino],
            canal=rng.choice(CANALES, p=CANAL_WEIGHTS), tipologia="rafaga_cuenta_nueva", es_ilicito=True,
            caso_id=case_id,
        )
    return batch, {objetivo}, objetivo, nueva_apertura


def inject_monto_inusual(rng: np.random.Generator, ids: list[str], banco_map: dict, case_id: str) -> tuple[TransferBatch, set[str]]:
    batch = TransferBatch()
    cuenta = str(rng.choice(ids))
    contraparte = str(rng.choice([a for a in ids if a != cuenta]))
    n_eventos = int(rng.integers(1, 3))
    baseline = rng.uniform(80_000, 350_000)
    for _ in range(n_eventos):
        multiplicador = rng.uniform(15, 50)
        monto = round(float(np.clip(baseline * multiplicador, 3_000_000, 80_000_000)), 0)
        ts = _sample_timestamp(rng, 0, SIMULATION_DAYS)
        if rng.random() < 0.5:
            origen, destino = cuenta, contraparte
        else:
            origen, destino = contraparte, cuenta
        batch.add(
            origen=origen, destino=destino, monto_clp=monto, timestamp=ts,
            banco_origen=banco_map[origen], banco_destino=banco_map[destino],
            canal=rng.choice(CANALES, p=CANAL_WEIGHTS), tipologia="monto_inusual", es_ilicito=True,
            caso_id=case_id,
        )
    return batch, {cuenta}


def generate(
    seed: int = SEED,
    n_accounts: int = N_ACCOUNTS,
    n_normal_transfers: int = N_NORMAL_TRANSFERS,
    n_pitufeo_cases: int = N_PITUFEO_CASES,
    n_puente_cases: int = N_PUENTE_CASES,
    n_rafaga_cases: int = N_RAFAGA_CASES,
    n_inusual_cases: int = N_INUSUAL_CASES,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Genera cuentas, transferencias y la tabla de verdad terreno (evaluacion).

    Los parametros de tamanio tienen como default las constantes del modulo
    (el dataset "de verdad" del proyecto); se pueden reducir para pruebas
    unitarias rapidas sin duplicar la logica de inyeccion de tipologias.

    Retorna (accounts_df, transfers_df, ground_truth_df).
    """
    rng = np.random.default_rng(seed)
    accounts = generate_accounts(rng, n_accounts)
    ids = accounts["account_id"].to_list()
    banco_map = dict(zip(ids, accounts["banco"].to_list()))

    all_rows: list[dict] = []
    etiquetas: dict[str, str] = {}
    aperturas_override: dict[str, datetime] = {}

    normales = generate_normal_transfers(rng, accounts, n_normal_transfers)
    all_rows.extend(normales.rows)

    for i in range(n_pitufeo_cases):
        batch, cuentas = inject_pitufeo(rng, ids, banco_map, f"PITUFEO-{i:03d}")
        all_rows.extend(batch.rows)
        for c in cuentas:
            etiquetas[c] = "pitufeo"

    for i in range(n_puente_cases):
        batch, cuentas = inject_cuenta_puente(rng, ids, banco_map, f"PUENTE-{i:03d}")
        all_rows.extend(batch.rows)
        for c in cuentas:
            etiquetas[c] = "cuenta_puente"

    for i in range(n_rafaga_cases):
        batch, cuentas, objetivo, nueva_apertura = inject_rafaga_cuenta_nueva(rng, ids, banco_map, f"RAFAGA-{i:03d}")
        all_rows.extend(batch.rows)
        for c in cuentas:
            etiquetas[c] = "rafaga_cuenta_nueva"
        aperturas_override[objetivo] = nueva_apertura

    for i in range(n_inusual_cases):
        batch, cuentas = inject_monto_inusual(rng, ids, banco_map, f"INUSUAL-{i:03d}")
        all_rows.extend(batch.rows)
        for c in cuentas:
            etiquetas.setdefault(c, "monto_inusual")

    transfers = pl.DataFrame(all_rows, infer_schema_length=None).sort("timestamp")
    transfer_ids = [f"TEF{idx:07d}" for idx in range(transfers.height)]
    transfers = transfers.with_columns(pl.Series("transfer_id", transfer_ids)).select(
        ["transfer_id", "timestamp", "origen", "destino", "monto_clp", "banco_origen",
         "banco_destino", "canal", "tipologia", "es_ilicito", "caso_id"]
    )

    if aperturas_override:
        override_ids = list(aperturas_override.keys())
        override_fechas = list(aperturas_override.values())
        override_df = pl.DataFrame({"account_id": override_ids, "fecha_apertura_nueva": override_fechas})
        accounts = accounts.join(override_df, on="account_id", how="left").with_columns(
            pl.when(pl.col("fecha_apertura_nueva").is_not_null())
            .then(pl.col("fecha_apertura_nueva"))
            .otherwise(pl.col("fecha_apertura"))
            .alias("fecha_apertura")
        ).drop("fecha_apertura_nueva")

    ground_truth = pl.DataFrame(
        {
            "account_id": list(etiquetas.keys()),
            "tipologia_real": list(etiquetas.values()),
        }
    ).with_columns(pl.lit(True).alias("es_cuenta_ilicita"))

    return accounts, transfers, ground_truth


def build_od_bank_matrix(transfers: pl.DataFrame) -> pl.DataFrame:
    """Matriz origen-destino agregada a nivel de banco (conteo y monto total)."""
    return (
        transfers.group_by(["banco_origen", "banco_destino"])
        .agg(pl.len().alias("n_transferencias"), pl.col("monto_clp").sum().alias("monto_total_clp"))
        .sort(["banco_origen", "banco_destino"])
    )


def main():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    accounts, transfers, ground_truth = generate()
    od_bancos = build_od_bank_matrix(transfers)

    accounts.write_parquet(OUT_DIR / "accounts.parquet")
    transfers.write_parquet(OUT_DIR / "transfers.parquet")
    ground_truth.write_parquet(OUT_DIR / "ground_truth.parquet")
    od_bancos.write_csv(OUT_DIR / "matriz_od_bancos.csv")
    transfers.head(2000).write_csv(OUT_DIR / "transfers_sample.csv")

    print(f"Cuentas generadas:        {accounts.height:,}")
    print(f"Transferencias generadas: {transfers.height:,}")
    print(f"Cuentas etiquetadas (verdad terreno, solo evaluacion): {ground_truth.height:,}")
    print(transfers.group_by("tipologia").agg(pl.len().alias("n")).sort("n", descending=True))
    print(f"\nArchivos escritos en: {OUT_DIR}")


if __name__ == "__main__":
    main()
