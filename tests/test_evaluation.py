"""Smoke-Tests: DM-Test und RMSE-Sanity fuer evaluation."""
import numpy as np
import pandas as pd
import pytest

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.evaluation import diebold_mariano, rolling_origin


# ── Diebold-Mariano ───────────────────────────────────────────────────────────

def test_dm_perfect_model():
    """Perfektes Modell (e_mod=0) sollte positive DM-Statistik haben."""
    np.random.seed(0)
    e_rw  = np.random.randn(36)
    e_mod = np.zeros(36)
    dm, pv = diebold_mariano(e_rw, e_mod, h=1)
    assert dm > 0, "DM-Stat sollte positiv sein wenn Modell perfekt"
    assert 0 <= pv <= 1, f"p-Wert ausserhalb [0,1]: {pv}"


def test_dm_rw_is_rw():
    """Identische Fehler (Modell = RW) sollen DM ≈ 0 ergeben."""
    np.random.seed(1)
    e_rw = np.random.randn(36)
    dm, pv = diebold_mariano(e_rw, e_rw, h=1)
    assert abs(dm) < 1e-10, f"DM sollte 0 sein bei gleichen Fehlern, got {dm}"


def test_dm_worse_model():
    """Schlechteres Modell (groessere Fehler) soll negative DM-Statistik haben."""
    np.random.seed(2)
    e_rw  = np.random.randn(36) * 1.0
    e_mod = np.random.randn(36) * 2.0   # groessere Fehler
    dm, pv = diebold_mariano(e_rw, e_mod, h=1)
    assert dm < 0, "DM-Stat sollte negativ sein wenn Modell schlechter als RW"
    assert 0 <= pv <= 1


def test_dm_p_value_range():
    """p-Wert muss in [0,1] liegen."""
    e1 = np.linspace(-1, 1, 36)
    e2 = np.linspace(-2, 2, 36)
    _, pv = diebold_mariano(e1, e2, h=1)
    assert 0.0 <= pv <= 1.0


# ── Rolling-Origin ────────────────────────────────────────────────────────────

def test_rolling_origin_length():
    """Rolling-Origin soll len(y) - start Prognosen liefern."""
    np.random.seed(3)
    n   = 50; start = 30
    X   = pd.DataFrame(np.random.randn(n, 3))
    y   = pd.Series(np.random.randn(n))
    from sklearn.linear_model import LinearRegression
    preds = rolling_origin(lambda: LinearRegression(), X, y, start)
    assert len(preds) == n - start, \
        f"Erwartet {n - start} Prognosen, erhalten {len(preds)}"


def test_rolling_origin_index_alignment():
    """Prognose-Index soll dem y-Index ab start entsprechen."""
    np.random.seed(4)
    idx = pd.date_range("2020-01", periods=40, freq="MS")
    X   = pd.DataFrame(np.random.randn(40, 2), index=idx)
    y   = pd.Series(np.random.randn(40), index=idx)
    from sklearn.linear_model import LinearRegression
    start = 25
    preds = rolling_origin(lambda: LinearRegression(), X, y, start)
    expected_idx = idx[start:]
    pd.testing.assert_index_equal(preds.index, expected_idx)


def test_rolling_origin_no_lookahead():
    """Prognose an Zeitpunkt t darf nur Daten bis t-1 nutzen (kein Look-ahead).

    Wir setzen y[0:start] = 0, y[start:] = 1. Ein korrektes Expanding-Window-Modell
    trainiert auf [0:start] und sagt fuer t=start einen Wert nahe 0 voraus,
    nicht den wahren Wert 1.
    """
    np.random.seed(5)
    n = 60; start = 40
    X = pd.DataFrame(np.random.randn(n, 1))
    y = pd.Series([0.0] * start + [1.0] * (n - start))
    from sklearn.linear_model import LinearRegression
    preds = rolling_origin(lambda: LinearRegression(), X, y, start)
    # Erste Prognose (trainiert nur auf Nullen) soll nahe 0 sein, nicht 1
    first_pred = preds.iloc[0]
    assert abs(first_pred) < 0.5, \
        f"Look-ahead-Verdacht: erste Prognose = {first_pred:.3f} (erwartet ≈ 0)"
