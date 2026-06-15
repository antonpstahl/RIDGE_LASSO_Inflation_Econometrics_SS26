"""Smoke-Tests: Schätzer-Sanity fuer AdaptiveLasso."""
import numpy as np
import pytest

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.models import AdaptiveLasso


@pytest.fixture
def xy_small():
    np.random.seed(42)
    n, p = 80, 10
    X = np.random.randn(n, p)
    y = X[:, 0] * 2.0 + X[:, 2] * -1.5 + np.random.randn(n) * 0.5
    return X, y


def test_adaptive_lasso_fit_predict_shape(xy_small):
    X, y = xy_small
    model = AdaptiveLasso(alphas=np.logspace(-3, 1, 10), cv=3).fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(y),), "predict() hat falsche Shape"


def test_adaptive_lasso_coef_shape(xy_small):
    X, y = xy_small
    model = AdaptiveLasso(alphas=np.logspace(-3, 1, 10), cv=3).fit(X, y)
    assert model.coef_.shape == (X.shape[1],), "coef_ hat falsche Shape"


def test_adaptive_lasso_deterministic(xy_small):
    """Zwei Fits mit gleichem Seed liefern identische Koeffizienten."""
    X, y = xy_small
    np.random.seed(7)
    m1 = AdaptiveLasso(alphas=np.logspace(-3, 1, 10), cv=3).fit(X, y)
    np.random.seed(7)
    m2 = AdaptiveLasso(alphas=np.logspace(-3, 1, 10), cv=3).fit(X, y)
    np.testing.assert_array_equal(
        m1.coef_, m2.coef_,
        err_msg="AdaptiveLasso ist nicht deterministisch",
    )


def test_adaptive_lasso_sparsity(xy_small):
    """Adaptive LASSO soll bei starker Regularisierung Sparsität erzeugen."""
    X, y = xy_small
    model = AdaptiveLasso(alphas=[1.0, 10.0, 100.0], cv=3).fit(X, y)
    n_nonzero = int(np.sum(model.coef_ != 0))
    assert n_nonzero < X.shape[1], \
        "Adaptive LASSO hat bei hohem α keine Koeffizienten auf null gesetzt"


def test_adaptive_lasso_alpha_attr(xy_small):
    X, y = xy_small
    model = AdaptiveLasso(alphas=np.logspace(-3, 1, 5), cv=3).fit(X, y)
    assert hasattr(model, "alpha_"), "alpha_-Attribut fehlt nach fit()"
    assert model.alpha_ > 0, "alpha_ muss positiv sein"
