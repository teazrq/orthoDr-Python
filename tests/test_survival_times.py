import pickle

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from orthodr import CP_SIR, orthoDr_surv, view_dr_surv


def test_large_integer_follow_up_preserves_risk_sets_and_event_labels():
    x = np.array([[-1.], [1.], [-1.], [1.]])
    times = np.arange(4, dtype=np.int64) + 1_700_000_000_000_000_000
    event = [1, 0, 1, 0]
    fit = orthoDr_surv(x, times, event, B_initial=[[1.]], keep_data=True, maxitr=0)
    restored = pickle.loads(pickle.dumps(fit))
    pred = restored.predict([[0.]])
    assert pred["timepoints"].dtype == times.dtype
    assert_array_equal(pred["timepoints"], times[[0, 2]])
    np.testing.assert_allclose(pred["surv"][:, 0], [.75, .375], rtol=1e-14)
    small = CP_SIR(x, np.arange(4), event)
    shifted = CP_SIR(x, times, event)
    assert_array_equal(shifted["values"], small["values"])


@pytest.mark.parametrize("row", [1, 2])
@pytest.mark.parametrize("missing", [np.nan, np.inf, "mask"])
def test_survival_times_report_original_event_or_censor_position(row, missing):
    x = np.array([[-1.], [1.], [-1.], [1.]])
    times = np.array([0., 2., -1., 0.])
    event = [1, 0, 1, 0]
    fit = orthoDr_surv(x, times, event, maxitr=0, keep_data=True)
    if missing == "mask":
        bad = np.ma.array(times, mask=np.arange(len(times)) == row)
    else:
        bad = times.copy()
        bad[row] = missing
    expected = f"0-based observation rows {row}"
    with pytest.raises(ValueError, match=expected):
        CP_SIR(x, bad, event)
    with pytest.raises(ValueError, match=expected):
        orthoDr_surv(x, bad, event, B_initial=np.ones((1, 1)), maxitr=0)
    for kind in ("2D", "3D"):
        with pytest.raises(ValueError, match=expected):
            view_dr_surv(x, bad, event, B=[1.], FUN="identity", type=kind)

    fit["y"] = bad
    restored = pickle.loads(pickle.dumps(fit))
    with pytest.raises(ValueError, match=expected + ".*Correct the original follow-up times and refit"):
        restored.predict(x[:1])


@pytest.mark.parametrize("kind", ["2D", "3D"])
def test_survival_plot_validates_transformed_times_before_drawing(kind):
    x = np.arange(4.).reshape(-1, 1)
    times = np.array([-1., 0., 0., 1.])
    event = [1, 0, 1, 0]
    calls = []

    def masked_transform(values):
        calls.append(values)
        return np.ma.array(values, mask=[False, False, True, False])

    with pytest.raises(ValueError, match=r"FUN\(y\).*observation rows 2"):
        view_dr_surv(x, times, event, B=[1.], FUN=masked_transform, type=kind)
    assert len(calls) == 1
    assert_array_equal(calls[0], times)
    with pytest.raises(ValueError, match=r"FUN\(y\).*one value per observation"):
        view_dr_surv(x, times, event, B=[1.], FUN=lambda y: y[:-1], type=kind)

    pytest.importorskip("matplotlib")
    ax = view_dr_surv(x, np.ma.array(times), event, B=[1.], FUN="identity", type=kind)
    assert ax.name == ("rectilinear" if kind == "2D" else "3d")
