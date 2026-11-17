"""Independent public behavior checks from the pinned R wrapper audit."""
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose

import orthodr.models as models


def test_nearly_parallel_mixed_initializer_is_skipped(monkeypatch):
    sir = np.eye(3)
    angle = 1e-9
    save = np.array([[np.cos(angle), -np.sin(angle), 0.],
                     [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    monkeypatch.setattr(models, "regression_initializers", lambda *a: dict(sir=sir, save=save, phd=sir))
    objectives = []

    def objective(B, *args):
        objectives.append(B.copy())
        return 1.

    def solver(B, *args):
        return dict(B=B, fn=1., itr=0, converge=True)

    monkeypatch.setattr(models, "_native", lambda: SimpleNamespace(sir_solver=solver, sir_init=objective))
    rng = np.random.default_rng(7)
    x = rng.normal(size=(40, 3))
    fit = models.orthoDr_reg(x, x[:, 0] + rng.normal(size=40), ndr=2)
    assert len(objectives) == 3
    assert_allclose(fit.B, np.eye(3)[:, :2])


@pytest.mark.parametrize("function", ["reg", "surv", "pdose"])
def test_invalid_keep_data_is_rejected_before_native_work(monkeypatch, function):
    monkeypatch.setattr(models, "_native", lambda: pytest.fail("native work must not start"))
    x = np.arange(12.).reshape(6, 2)
    y = np.arange(1., 7.)
    with pytest.raises(ValueError, match="keep_data"):
        if function == "reg":
            models.orthoDr_reg(x, y, keep_data="yes")
        elif function == "surv":
            models.orthoDr_surv(x, y, np.ones(6), keep_data="yes")
        else:
            models.orthoDr_pdose(x, y, y, keep_data="yes")
