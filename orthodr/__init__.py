"""Semiparametric dimension reduction for NumPy data, without an R runtime."""

from .models import FitResult, orthoDr_reg, orthoDr_surv, orthoDr_pdose, predict
from .initializers import CP_SIR, pSAVE, hMave
from .optim import ortho_optim
from ._utils import silverman, distance, dist_cross, kernel_weight
from .plotting import view_dr_surv
from .datasets import load_skcm

__version__ = "0.1.0"

__all__ = [
    "FitResult", "orthoDr_reg", "orthoDr_surv", "orthoDr_pdose", "predict",
    "CP_SIR", "pSAVE", "hMave", "ortho_optim", "silverman", "distance",
    "dist_cross", "kernel_weight", "view_dr_surv", "load_skcm",
]
