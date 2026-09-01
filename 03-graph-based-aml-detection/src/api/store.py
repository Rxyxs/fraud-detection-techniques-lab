"""Persistencia del historial de decisiones y alertas de la API en SQLite.

Se eligio SQLite (via el modulo estandar ``sqlite3``, sin dependencias
adicionales) sobre DuckDB porque el caso de uso -- registrar transacciones
puntuales scoreadas y decisiones de analista, una fila a la vez -- es
escritura transaccional fila-por-fila (el punto fuerte de SQLite), no
analitica columnar sobre grandes volumenes (el punto fuerte de DuckDB). El
archivo de la base vive en ``outputs/`` (regenerable, gitignored), igual que
el resto de los artefactos del pipeline.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

DB_PATH = Path(__file__).resolve().parents[2] / "outputs" / "decisiones_analista.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS historial_scoring (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transfer_id TEXT NOT NULL,
    origen TEXT NOT NULL,
    destino TEXT NOT NULL,
    monto_clp REAL NOT NULL,
    timestamp_transaccion TEXT NOT NULL,
    score_anomalia REAL NOT NULL,
    es_alerta INTEGER NOT NULL,
    scoreado_en TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisiones_analista (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transfer_id TEXT NOT NULL,
    analista TEXT NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('confirmado_ilicito', 'falso_positivo', 'pendiente_revision')),
    notas TEXT,
    decidido_en TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_scoring_transfer_id ON historial_scoring(transfer_id);
CREATE INDEX IF NOT EXISTS idx_decisiones_transfer_id ON decisiones_analista(transfer_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_connection(db_path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path | None = None) -> None:
    with get_connection(db_path or DB_PATH) as conn:
        conn.executescript(_SCHEMA)


def guardar_scoring(
    transfer_id: str,
    origen: str,
    destino: str,
    monto_clp: float,
    timestamp_transaccion: str,
    score_anomalia: float,
    es_alerta: bool,
    db_path: Path | None = None,
) -> int:
    with get_connection(db_path or DB_PATH) as conn:
        cur = conn.execute(
            """INSERT INTO historial_scoring
               (transfer_id, origen, destino, monto_clp, timestamp_transaccion,
                score_anomalia, es_alerta, scoreado_en)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (transfer_id, origen, destino, monto_clp, timestamp_transaccion,
             score_anomalia, int(es_alerta), _now()),
        )
        return cur.lastrowid


def guardar_decision(
    transfer_id: str,
    analista: str,
    decision: str,
    notas: str | None = None,
    db_path: Path | None = None,
) -> int:
    with get_connection(db_path or DB_PATH) as conn:
        cur = conn.execute(
            """INSERT INTO decisiones_analista (transfer_id, analista, decision, notas, decidido_en)
               VALUES (?, ?, ?, ?, ?)""",
            (transfer_id, analista, decision, notas, _now()),
        )
        return cur.lastrowid


def listar_alertas(limite: int = 100, db_path: Path | None = None) -> list[dict]:
    """Historial de transacciones scoreadas como alerta, con la ultima
    decision de analista registrada (si existe), mas reciente primero."""
    with get_connection(db_path or DB_PATH) as conn:
        filas = conn.execute(
            """
            SELECT
                s.transfer_id, s.origen, s.destino, s.monto_clp,
                s.timestamp_transaccion, s.score_anomalia, s.es_alerta, s.scoreado_en,
                d.analista, d.decision, d.notas, d.decidido_en
            FROM historial_scoring s
            LEFT JOIN (
                SELECT transfer_id, analista, decision, notas, decidido_en,
                       ROW_NUMBER() OVER (PARTITION BY transfer_id ORDER BY id DESC) AS rn
                FROM decisiones_analista
            ) d ON d.transfer_id = s.transfer_id AND d.rn = 1
            WHERE s.es_alerta = 1
            ORDER BY s.id DESC
            LIMIT ?
            """,
            (limite,),
        ).fetchall()
        return [dict(fila) for fila in filas]
