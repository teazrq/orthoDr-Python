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
