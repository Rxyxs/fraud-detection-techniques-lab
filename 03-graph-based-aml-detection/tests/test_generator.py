import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.synthetic_generator import build_od_bank_matrix, generate


def _small_dataset(seed=1):
    return generate(
        seed=seed, n_accounts=150, n_normal_transfers=1500,
        n_pitufeo_cases=3, n_puente_cases=3, n_rafaga_cases=3, n_inusual_cases=3,
    )


def test_schema_and_no_nulls():
    accounts, transfers, ground_truth = _small_dataset()

    assert set(accounts.columns) >= {"account_id", "tipo_cliente", "banco", "region", "fecha_apertura"}
    assert set(transfers.columns) >= {
        "transfer_id", "timestamp", "origen", "destino", "monto_clp", "tipologia", "es_ilicito",
    }
    for col in ["account_id", "tipo_cliente", "banco", "fecha_apertura"]:
        assert accounts[col].null_count() == 0
    for col in ["transfer_id", "origen", "destino", "monto_clp", "timestamp"]:
        assert transfers[col].null_count() == 0

    assert accounts["account_id"].n_unique() == accounts.height
    assert transfers["transfer_id"].n_unique() == transfers.height


def test_typologies_are_injected():
    _, transfers, ground_truth = _small_dataset()
    tipologias = set(transfers["tipologia"].unique().to_list())
    assert {"normal", "pitufeo", "cuenta_puente", "rafaga_cuenta_nueva", "monto_inusual"} <= tipologias
    assert ground_truth.height > 0
    assert set(ground_truth["tipologia_real"].unique().to_list()) <= {
        "pitufeo", "cuenta_puente", "rafaga_cuenta_nueva", "monto_inusual",
    }


def test_no_self_loops():
    _, transfers, _ = _small_dataset()
    assert transfers.filter(transfers["origen"] == transfers["destino"]).height == 0


def test_amounts_are_positive():
    _, transfers, _ = _small_dataset()
    assert (transfers["monto_clp"] > 0).all()


def test_reproducible_with_same_seed():
    _, t1, _ = _small_dataset(seed=7)
    _, t2, _ = _small_dataset(seed=7)
    assert t1.height == t2.height
    assert t1["monto_clp"].sum() == t2["monto_clp"].sum()


def test_od_bank_matrix():
    _, transfers, _ = _small_dataset()
    od = build_od_bank_matrix(transfers)
    assert {"banco_origen", "banco_destino", "n_transferencias", "monto_total_clp"} <= set(od.columns)
    assert od["n_transferencias"].sum() == transfers.height
