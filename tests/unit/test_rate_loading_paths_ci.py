from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from kimeco.rate_coef import RateCo


def _make_rateco(rows, pres, temp, kin_id=0):
    """Build a bare RateCo wired only for load_rates_from_db."""
    rc = RateCo.__new__(RateCo)
    rc.db = cast(Any, SimpleNamespace(
        get_rates_for_kin_id=lambda table, kin_id: rows))
    rc.sop = cast(Any, SimpleNamespace(
        pes_ids=[0],
        species_names_in_pes=lambda pes: ['A', 'B']))
    rc.pres = pres
    rc.temp = temp
    rc.id = kin_id
    return rc


def test_load_rates_from_db_populates_attrs() -> None:
    k_value = 1.23e12
    rows = [
        # (p, t, pes_id, from_name, to_name, k_value)
        (1.0, 300.0, 0, 'A', 'B', k_value),      # valid A->B cell
        (0.5, 300.0, 0, 'A', 'B', 9.9),          # off-grid pressure -> skipped
        (1.0, 300.0, 0, 'A', 'X', 8.8),          # unknown species -> skipped
    ]
    rc = _make_rateco(rows, pres=[1.0], temp=[300.0])

    rc.load_rates_from_db(table='G0000')

    assert rc.tbl_map_by_pes[0] == {'A': 0, 'B': 1}
    arr = rc.rc_by_pes[0]
    assert arr.shape == (1, 1, 2, 2)
    # Valid entry landed at [p_idx=0, t_idx=0, from=A(0), to=B(1)].
    assert arr[0, 0, 0, 1] == k_value
    # Everything else (incl. off-grid / unknown-species rows) stays zero.
    total_nonzero = int((arr != 0).sum())
    assert total_nonzero == 1


def test_load_rates_from_db_empty_rows_yields_zero_arrays() -> None:
    rc = _make_rateco(rows=[], pres=[1.0, 2.0], temp=[300.0, 400.0])

    rc.load_rates_from_db(table='G0000')

    assert rc.tbl_map_by_pes[0] == {'A': 0, 'B': 1}
    arr = rc.rc_by_pes[0]
    assert arr.shape == (2, 2, 2, 2)
    assert int((arr != 0).sum()) == 0
