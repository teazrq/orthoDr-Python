import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from orthodr.datasets import load_skcm
from orthodr.plotting import _weighted_km_surface, view_dr_surv


def test_skcm_dimensions_identifiers_missingness_and_r_provenance():
    data = load_skcm()
    clinical, genes, p = data["clinical"], data["genes"], data["provenance"]
    assert clinical.shape == (469, 4) and genes.shape == (469, 20)
    assert_array_equal(data["clinical_columns"], ["Time", "Censor", "Gender", "Age"])
    assert_array_equal(data["gene_names"][:4], ["TYRP1", "OCA2", "TYR", "SLC45A2"])
    assert data["subject_ids"][0] == "TCGA-BF-A1PU"
    assert data["subject_ids"][-1] == "TCGA-ER-A2NB"
    assert len(np.unique(data["subject_ids"])) == 469
    assert p["source_commit"] == "d99bfa8" and p["identifiers_aligned"]
    assert p["source_md5"] == ["2ade974f3b216a4018323cb4644ee415", "2f7194da903be89d3fb4551dc2627b17"]
    assert "gene_subject_ids" not in data
    assert_array_equal(np.isnan(clinical).sum(axis=0), [0, 0, 0, 8])
    assert not np.isnan(genes).any()
    assert (clinical[:, 1] == 1).sum() == p["event_count"] == 156
    assert_array_equal(data["subject_ids"][np.isnan(clinical[:, 3])], p["missing_age_subject_ids"])
    assert_array_equal(np.nansum(clinical, axis=0), p["clinical_sums"])
    assert_array_equal(np.nansum(clinical ** 2, axis=0), p["clinical_sum_squares"])
    assert_allclose(genes.sum(axis=0), p["gene_sums"], atol=5e-11, rtol=0)
    assert_allclose((genes ** 2).sum(axis=0), p["gene_sum_squares"], atol=6e-10, rtol=0)
    assert_allclose(genes[0], p["first_genes"], atol=5e-13, rtol=0)
    assert_allclose(genes[-1], p["last_genes"], atol=5e-13, rtol=0)
    assert_array_equal(clinical[0], [387, 0, 0, 46])
    assert_array_equal(clinical[-1], [486, 0, 1, 57])


def test_skcm_returns_independent_arrays():
    first = load_skcm()
    first["clinical"][0, 0] = -1
    first["genes"][0, 0] = -1
    first["subject_ids"][0] = "changed"
    second = load_skcm()
    assert second["clinical"][0, 0] == 387
    assert second["genes"][0, 0] > 0
    assert second["subject_ids"][0] == "TCGA-BF-A1PU"


def test_weighted_km_uses_complete_tied_time_risk_sets():
    index = np.array([0., 1., 2., 3.])
    times = np.array([1., 2., 2., 4.])
    events = np.array([1., 1., 0., 1.])
    xgrid, timegrid, bw = np.array([0., 1.]), np.array([0., 1., 2., 3., 4., 5.]), 2.
    surface = _weighted_km_surface(index, times, events, xgrid, timegrid, bw)
    for i, target in enumerate(xgrid):
        w = np.exp(-.5 * ((target - index) / bw) ** 2)
        after_first = 1 - w[0] / w.sum()
        after_tie = after_first * (1 - w[1] / w[1:].sum())
        assert_allclose(surface[i], [1, after_first, after_tie, after_tie, 0, 0], atol=1e-15)
    permutation = np.array([2, 0, 3, 1])
    permuted = _weighted_km_surface(index[permutation], times[permutation], events[permutation], xgrid, timegrid, bw)
    assert_allclose(permuted, surface, atol=1e-15)
    assert np.all(np.diff(surface, axis=1) <= 0)


@pytest.fixture
def surface_axes(monkeypatch):
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure
    ax = Figure().add_subplot(projection="3d")
    captured = {}

    def capture(x, time, survival, **kwargs):
        captured.update(x=x.copy(), time=time.copy(), survival=survival.copy(), **kwargs)

    monkeypatch.setattr(ax, "plot_surface", capture)
    return ax, captured


def _independent_km(index, times, events, xgrid, timegrid, bw):
    weights = np.exp(-.5 * ((xgrid[:, None] - index) / bw) ** 2)
    result = np.ones((len(xgrid), len(timegrid)))
    for time in np.unique(times[events == 1]):
        deaths = weights[:, (times == time) & (events == 1)].sum(axis=1)
        risk = weights[:, times >= time].sum(axis=1)
        result[:, timegrid >= time] *= (1 - deaths / risk)[:, None]
    return result


def test_surface_keeps_failures_with_underflowed_absolute_weights(surface_axes):
    ax, actual = surface_axes
    index = np.array([0., 0., 100., 100., 100., 100.])
    times = np.array([1., 1., 2., 3., 4., 4.])
    events = np.array([1, 0, 1, 1, 1, 0])
    view_dr_surv(index[:, None], times, events, [1.], bw=.001,
                 FUN="identity", type="3D", ax=ax)
    assert_allclose(actual["survival"][0], [.5, .375, .25, .125], rtol=1e-14, atol=0)
    assert_allclose(actual["survival"][-1], [1., .75, .5, .25], rtol=1e-14, atol=0)
    assert np.all(np.diff(actual["survival"], axis=1) <= 0)


@pytest.mark.parametrize("n, offset", [(1000, 1), (1001, 0)])
def test_surface_uses_pooled_empirical_percentiles_and_every_event(surface_axes, n, offset):
    ax, actual = surface_axes
    y = (np.arange(1, n + 1, dtype=float) / (n + 1)) ** 2
    index = np.linspace(-2, 2, n)
    events = np.zeros(n)
    events[[0, 333, 777, n - 1]] = 1
    # For n=1000 the kth percentile is observation 10*k. For n=1001 it
    # is observation 10*k+1. These exact boundaries must not round upward.
    selected = np.unique(np.r_[0, np.arange(10, 1001, 10) - offset, 333, 777])
    expected_time = y[selected]
    permutation = np.random.default_rng(91).permutation(n)
    result = view_dr_surv(index[permutation, None], y[permutation], events[permutation],
                          [1.], bw=.8, FUN=None, type="3D", ax=ax)
    assert result is ax
    xgrid = np.linspace(index.min(), index.max(), 100)
    assert_array_equal(actual["x"], np.broadcast_to(xgrid[:, None], actual["x"].shape))
    assert_array_equal(actual["time"], np.broadcast_to(expected_time, actual["time"].shape))
    assert actual["rcount"] == 100 and actual["ccount"] == len(expected_time)
    assert len(expected_time) < len(y)
    assert np.isin(expected_time, y).all()
    assert expected_time[0] == y.min() and expected_time[-1] == y.max()
    assert np.isin(y[events == 1], expected_time).all()
    expected = _independent_km(index, y, events, xgrid, expected_time, .8)
    assert_allclose(actual["survival"], expected, atol=1e-14, rtol=0)


@pytest.mark.parametrize("FUN, transform, units", [
    (None, lambda value: value, 1.),
    ("identity", lambda value: value, 365.25),
    ("log", np.log, 1.),
    ("log", np.log, 365.25),
    ("log10", np.log10, 1.),
    ("log2", np.log2, 1.),
    ("sqrt", np.sqrt, 1.),
])
def test_surface_knots_and_heights_follow_monotone_time_transforms(surface_axes, FUN, transform, units):
    ax, actual = surface_axes
    y = np.array([.8, .2, .5, .2, .9, .5, .35, .5])
    index = np.linspace(-1, 1, len(y))
    events = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    view_dr_surv(index[:, None], units * y, events, [1.], bw=.6,
                 FUN=FUN, type="3D", ax=ax)
    expected_time = np.unique(y)
    assert_allclose(actual["time"][0], transform(units * expected_time), atol=0, rtol=0)
    xgrid = np.linspace(index.min(), index.max(), 100)
    expected = _independent_km(index, y, events, xgrid, expected_time, .6)
    assert_allclose(actual["survival"], expected, atol=1e-14, rtol=0)
    assert np.all(np.diff(actual["survival"], axis=1) <= 0)


def test_surface_calls_full_vector_transform_once_and_keeps_its_coordinates(surface_axes):
    ax, actual = surface_axes
    y = np.linspace(.1, .9, 1000)
    original = y.copy()
    calls = []

    def transform(values):
        calls.append(values.copy())
        values -= values.mean()
        return values

    events = np.zeros(len(y))
    events[333] = 1
    view_dr_surv(np.linspace(-1, 1, len(y))[:, None], y, events, [1.],
                 bw=.7, FUN=transform, type="3D", ax=ax)
    assert len(calls) == 1
    assert_array_equal(calls[0], original)
    assert_array_equal(y, original)
    selected = np.unique(np.r_[0, np.arange(9, 1000, 10), 333])
    assert_array_equal(actual["time"][0], original[selected] - original.mean())


@pytest.mark.parametrize("events", [[1, 0, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1]])
def test_all_tied_times_draw_the_actual_survival_profile(events):
    pytest.importorskip("matplotlib")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    index = np.array([-1., -.25, .5, 1.])
    y = np.full(len(index), .2)
    ax = view_dr_surv(index[:, None], y, events, [1.], bw=.7, FUN="log", type="3D")
    assert len(ax.lines) == 1 and len(ax.collections) == 0
    xgrid, timegrid, survival = ax.lines[0].get_data_3d()
    assert_array_equal(xgrid, np.linspace(index.min(), index.max(), 100))
    assert_array_equal(timegrid, np.full(100, np.log(.2)))
    expected = _independent_km(index, y, np.asarray(events), xgrid, [.2], .7)
    assert_allclose(survival, expected[:, 0], atol=1e-14, rtol=0)
    FigureCanvasAgg(ax.figure).draw()


def test_scatter_and_surface_return_axes_without_global_display(tmp_path):
    pytest.importorskip("matplotlib")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib import pyplot as plt
    before = plt.get_fignums()
    x = np.column_stack((np.linspace(-1, 1, 8), np.arange(8.)))
    y = np.array([1., 2., 2., 3., 4., 5., 6., 6.])
    censor = np.array([1, 0, 1, 1, 0, 1, 1, 0])
    ax = view_dr_surv(x, y, censor, [1., 0.], FUN=None)
    assert ax.get_xlabel() == "Reduced Direction" and ax.get_ylabel() == "Time"
    assert len(ax.collections[0].get_offsets()) == 5
    assert len(ax.collections[1].get_offsets()) == 3
    assert ax.get_legend() is not None
    FigureCanvasAgg(ax.figure).draw()
    ax.figure.savefig(tmp_path / "survival_scatter.png")
    surface = view_dr_surv(x, y, censor, [[1.], [0.]], type="3D", FUN="log")
    assert surface.name == "3d" and surface.get_zlabel() == "Survival"
    assert surface.get_zlim() == (0., 1.)
    FigureCanvasAgg(surface.figure).draw()
    surface.figure.savefig(tmp_path / "survival_surface.png")
    assert plt.get_fignums() == before
    assert (tmp_path / "survival_surface.png").stat().st_size > 1000


def test_survival_scatter_accepts_all_censored_boolean_status():
    pytest.importorskip("matplotlib")
    x = np.arange(4.).reshape(-1, 1)
    ax = view_dr_surv(x, np.arange(1., 5.), np.zeros(4, dtype=bool), [1.])
    assert len(ax.collections[0].get_offsets()) == 0
    assert len(ax.collections[1].get_offsets()) == 4


def test_display_time_rounding_preserves_last_failure():
    pytest.importorskip("matplotlib")
    x = np.arange(4.)[:, None]
    t = 2**52 + np.arange(1, 5)
    ax = view_dr_surv(x, t, [0, 0, 0, 1], [1.], bw=1., type="3D")
    _, _, survival = ax.lines[0].get_data_3d()
    assert_array_equal(survival, np.zeros(100))


@pytest.mark.parametrize("changes, match", [
    ({"B": None}, "B must be given"),
    ({"B": [1., 0., 0.]}, "one value"),
    ({"type": "4D"}, "type"),
    ({"type": "3D", "bw": 0}, "bw"),
    ({"y": np.array([0., 1., 2., 3.])}, "finite"),
    ({"censor": np.array([1, 2, 0, 0])}, "0/1"),
    ({"type": "3D", "B": [0., 0.]}, "varying"),
])
def test_plotting_validates_survival_and_direction_inputs(changes, match):
    args = dict(x=np.arange(8.).reshape(4, 2), y=np.arange(1., 5.),
                censor=np.array([1, 0, 1, 1]), B=[1., 0.])
    args.update(changes)
    with pytest.raises(ValueError, match=match):
        view_dr_surv(**args)
