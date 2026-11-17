# Python API

Look up array conventions, fitting arguments, returned diagnostics, and
numerical utilities for the GitHub source version **0.1.0**. For a first
analysis, start with [installation and regression](tutorials/get-started.md),
[survival](tutorials/survival.md), or
[personalized doses](tutorials/personalized-dose.md).

## Data and fitted objects

Use finite real NumPy-compatible arrays. Predictors have shape `(n, p)`;
responses, doses, rewards, and event indicators have shape `(n,)`.
Categorical encoding and missing-data handling are the caller's responsibility.
Apply any external preprocessing using training data only and reuse it for
new observations. Regression and dose fitting use predictors on their supplied
scale; survival fitting standardizes them internally.

The three model functions return `FitResult`, a dictionary with attribute
access and `transform`/`predict` methods. `B` is a `(p, ndr)` direction matrix,
`fn` is the returned objective, `itr` counts completed iterations, and
`converge` reports whether that method's stopping criterion was met. Both
dose methods have additional diagnostics described below. Regression and survival use
their original R stopping criteria, which can also stop on small changes.

`fit.transform(new_x)` returns `new_x @ fit.B`. Prediction requires
`keep_data=True` during fitting. `fit.predict(new_x)` returns a one-dimensional
array for regression and dose models, and a dictionary for survival models.
The top-level `predict(fit, new_x)` always returns a dictionary, with `pred`
for regression/dose or `surv` and `timepoints` for survival.

Fits saved with `pickle` retain their prediction state. This revision preserves
prediction for the earlier no-version pseudo-direct and existing direct-dose
objects. A fit saved by R is not a Python
pickle and cannot be loaded directly.

## Common fitting arguments

| Argument | Meaning |
| --- | --- |
| `ndr` | Positive integer number of directions. Regression and survival cap it at `p`; dose learning requires `ndr <= p`. |
| `B_initial` | Full-column-rank starting matrix of shape `(p, ndr)`, orthonormalized before fitting. Default: use the method's initializer. |
| `bw` | Positive kernel bandwidth. Default: `silverman(ndr, n)`. |
| `keep_data` | Retain training data for prediction. Default: `False`. |
| `control` | Dictionary of optimizer settings described below. |
| `maxitr` | Iteration budget, default 500. Regression allows at least five iterations. Both dose methods accept zero to return initialization. |
| `verbose` | Nonnegative integer verbosity level, default 0. |
| `ncore` | Number of native threads, default 1; 0 requests automatic detection. Both dose optimizers are serial. |

Default bandwidth is

$$
h=\left(\frac{4}{d+2}\right)^{1/(d+4)}n^{-1/(d+4)}.
$$

Projected coordinates are divided by their training sample standard deviations
inside the kernels. New observations use the training scales, including when
only one new row is supplied. Very distant regression/survival queries can
have all Gaussian weights underflow to zero; their predictions are `NaN` with
a warning.

`control` defaults are `rho=1e-4`, `eta=0.2`, `gamma=0.85`, `tau=1e-3`,
`epsilon=1e-6`, `btol=ftol=gtol=1e-6`. The first four govern line search and
trial steps; `epsilon` is the finite-difference increment; the last three are
tolerances for changes in directions, objective, and gradient. The source R
behavior of replacing invalid positive controls with defaults is retained for
the finite-difference solvers. Direct learning instead rejects invalid
`rho`, `eta`, `tau`, `btol`, `ftol`, and `gtol`; `rho` and `eta` must also be
below one. Its analytic, monotone solver does not use `gamma` or `epsilon`.
Pseudo-direct learning likewise rejects invalid controls; `gamma` may range
from zero through one in its nonmonotone search. Its analytic solver does not
use `epsilon`, and `gtol` applies to the normalized constrained gradient below.

## Regression

```text
orthoDr_reg(x, y, method="sir", ndr=2, B_initial=None, bw=None,
            keep_data=False, control=None, maxitr=500, verbose=0, ncore=1)
```

Choose from `sir`, `save`, `phd`, `local`, and `seff`. The first three solve
the corresponding semiparametric estimating-equation objectives; they are not
simply the sliced initializers. `local` uses local conditional-mean estimation,
and `seff` uses the efficient estimating equation.

The response must vary. Automatic initialization uses SIR/SAVE/residual-PHD
from the translated `dr` routines. It also considers the leading SIR and SAVE
directions jointly for `ndr=2`. One-direction LOCAL fits three starts and
chooses by training kernel R-squared. Other configurations select the initial
basis with the smallest starting objective. Automatic initialization requires
full centered predictor rank and at least two observations in each SAVE
response slice. All three candidates are computed even when the fitted method
is not SAVE, so small samples can fail this initialization requirement.
Supplying a full-column-rank `B_initial` bypasses the sliced initializers;
projected variances must still be positive.

## Survival

```text
orthoDr_surv(x, y, censor, method="dm", ndr=None, B_initial=None,
             bw=None, keep_data=False, control=None, maxitr=500,
             verbose=0, ncore=1)
```

`censor` contains 1 for an event and 0 for right censoring. At least one event
and positive predictor variances are required. `forward` uses one direction;
`dn` and `dm` default to two. Automatic initialization uses counting-process
SIR. A supplied `B_initial` is expressed in the internally standardized
predictor coordinates, matching R. The returned `B` is on the original scale.

Prediction returns a matrix of shape `(number_of_training_events, n_new)`.
Rows correspond to sorted observed event times, with repeats retained. Each
column is a kernel-weighted Kaplan-Meier curve, using complete tied risk sets.
The returned basis columns have unit length but need not be orthogonal after
conversion to the original scale.

## Personalized dose learning

```text
orthoDr_pdose(x, a, r, ndr=2, B_initial=None, bw=None, lambda_init=0.1,
              K=None, method="direct", keep_data=False, control=None,
              maxitr=500, verbose=0, ncore=1, lambda_reward=None,
              intercept=None, intercept_bounds=None)
```

Larger `r` is better. `a` must vary. Both dose methods use `pSAVE` when
`B_initial=None`. Supply a full-column-rank `(p, ndr)` matrix to use another
starting point, including a random one, and bypass `pSAVE`. The function
orthonormalizes the supplied basis before fitting. pSAVE is one initialization
option; final dose accuracy can vary with the starting point. A supplied basis
also permits fitting when pSAVE has no admissible dose split or informative
reward slices.
Use unit-length starting columns or an orthonormal basis. Extremely small
columns can be rejected as numerically rank deficient.

`K` is the number of candidate doses
on a grid spanning the observed range, default `floor(sqrt(n))`. Fractional
values are floored; `K=1` uses the minimum dose. Direct learning uses the grid
only during initialization. Pseudo-direct learning uses it to select targets
for the final ridge dose rule. Predictions from both methods are continuous.

Interpreting the rule causally requires consistency (the observed reward equals
the potential reward at the observed dose), no unmeasured dose-outcome
confounding given `x`, and adequate dose support in the relevant reduced
space. The function does not check these study assumptions; see
[Zhou, Zhu, and Zeng (2021)](https://doi.org/10.1093/biomet/asaa087) for the model
and identification conditions.

`lambda_init` is the Python spelling of R's `lambda` argument. It must be
positive. A vector invokes GCV. Direct learning uses it only for the initial
ridge rule; pseudo-direct learning uses it for the final ridge rule.
`lambda_reward=None` selects the nominal default 0.1 for direct learning.
Zero removes the penalty when `intercept=False`; a positive penalty is
required for direct `intercept=True`. Omit `lambda_reward` and
`intercept_bounds` for `method="pseudo_direct"`; its optional intercept is
described below.

`intercept=None` selects the R default, `False`, with rule
$d(x)=k_B(x)^\top w$. The vector $k_B(x)$ contains Gaussian kernel
comparisons with the projected training observations. A positive
`lambda_reward` shrinks this rule toward zero dose, so its effect depends on
the dose origin. For direct learning with `intercept=True`, centered kernel features make the
unpenalized intercept the mean training recommendation; the coefficients then
describe variation around that average. `intercept_bounds` is a finite
increasing pair, defaulting to `[min(a), max(a)]`. It bounds that mean, not
individual predictions. For intercept initialization, use a scalar `lambda_init` when custom bounds
constrain the mean grid target, except when all targets equal an attainable
constant. This preserves the R restriction on applying linear-smoother GCV
under an active intercept bound.

Dose kernel comparisons use the sample standard deviation of the observed
training doses for observed and candidate doses alike. Predictions are in the
original dose units. For direct learning, multiplying rewards by `c > 0` requires multiplying
`lambda_reward` by `c` to preserve the objective tradeoff. Multiplying dose
units by `s > 0` requires dividing it by `s**2`. In intercept mode, shifting
doses and intercept bounds together preserves the objective. These identities
do not guarantee identical optimizer trajectories with absolute tolerances.

The direct coefficient penalty is `lambda_reward * sum(W**2)`, not a squared RKHS norm.
Direction updates keep coefficients fixed while kernel centers and projection
scales change with `B`; this is the R package's joint finite-dimensional
formulation, which differs from a literal fixed-function direction update.
Coefficients are not refitted after stopping.

| Direct-fit field | Interpretation |
| --- | --- |
| `value`, `penalty`, `fn` | Training estimated reward, penalty contribution, and `-value + penalty` |
| `lambda.init`, `lambda.reward` | Selected initialization ridge value and supplied reward penalty |
| `gradient.norm` | Returned direction and coefficient residuals; intercept fits also give intercept and joint policy residuals |
| `stop.reason` | Gradient tolerance, iteration limit, stagnation, failed line search, or numerical failure |
| `intercept`, `intercept.bounds`, `kernel.mean` | Stored mean dose, its bounds, and training kernel centering values, for intercept fits |
| `objective.version` | `direct_joint_v1` or `direct_joint_intercept_v2` |

Access dotted keys directly, or replace dots with underscores for attributes.
Only the gradient-tolerance stop certifies direct numerical convergence:
both the direction and policy residuals must meet `control["gtol"]`.
`btol` and `ftol` detect unsuccessful stagnation. Failed searches preserve the
last completed sweep. The intercept solver retains the R spectral direction
step and its monotone line search. Training reward, numerical convergence,
and accuracy on new observations describe different properties of the fit.

Pseudo-direct learning fits the conditional reward surface, then constructs a
kernel ridge rule from grid-selected doses. It optimizes the mean squared
residual after centering rewards and dividing by their sample SD. This has
the same minimizers as raw RMSE, which is still returned as `fn`. Exactly
constant rewards are rejected because they identify neither directions nor
a preferred dose. The smoother includes each training observation itself;
small training error at a narrow bandwidth does not establish accurate decisions.

Its final ridge penalty is `lambda_init * W.T @ K_B @ W`, the squared RKHS norm
penalty. With `intercept=True`, the rule is $d(x)=\alpha+k_B(x)^\top W$.
The constant is unpenalized and unbounded. GCV accounts for that constant;
shifting the doses and grid shifts the fitted rule accordingly. The default
remains `False`. This intercept differs from direct learning's bounded,
centered average-dose parameter.

Reward-surface fitting uses bandwidth `bw` for dose and each standardized
projection. The final dose-grid smoother retains dose bandwidth `bw/sqrt(2)`
in training-dose SD units. Its log weights prevent undefined ratios, but a
grid cell far from the observations can still be dominated by one observation.

| Pseudo-direct field | Interpretation |
| --- | --- |
| `fn`, `objective.scaled`, `reward.scale` | Raw reward RMSE, normalized MSE, and training reward SD |
| `gradient.norm` | Scalar Frobenius norm of $G-BG^\top B$ for normalized MSE |
| `stop.reason` | `gradient_tolerance`, `stagnation`, `max_iterations`, `line_search_failed`, or `numerical_failure` |
| `lambda.init` | Selected ridge penalty for the final dose rule |
| `dose.bandwidth.scale` | Final dose-grid bandwidth in original dose units |
| `intercept` | Unbounded constant, present only when enabled |
| `objective.version` | `pseudo_mse_v1` or `pseudo_mse_intercept_v1` |

Only `gradient_tolerance` sets pseudo-direct `converge=True`. Failed searches
preserve the last accepted state. Stagnation requires small direction and
objective changes averaged over five accepted updates. Diagnostic scalars remain available with
`keep_data=False`; recomputing them requires the original training inputs.
See the runnable [pseudo-direct example](https://github.com/teazrq/orthoDr-Python/blob/main/examples/pseudo_direct.py).

## Initializers and general optimization

| Function | Result |
| --- | --- |
| `pSAVE(x, a, r, ndr=2, nslices0=2)` | `(p, ndr)` partial-SAVE basis |
| `CP_SIR(x, y, censor, bw=None)` | Dictionary of singular `values` and normalized direction `vectors` |
| `hMave(x, y, censor, m0, B0=None)` | Dictionary with `(p, m0)` basis `B` and the source hMAVE leave-one-out kernel score `cv` |
| `ortho_optim(B, fn, grad=None, *args, maximize=False, control=None, maxitr=500, verbose=0, **kwargs)` | Dictionary with `B`, `fn`, `fn_Seq`, `itr`, `converge`, and gradient `method` |

`pSAVE` now combines group SAVE information in common coordinates and returns
unit-norm directions in the original predictor units. These columns need not
be orthogonal; dose wrappers orthonormalize the starting basis in column order.
Tiny outcome slices add no directional information. Automatic starts for both
dose methods can change with this correction. The initializer still requires
adequate rank and informative slices; it does not guarantee subspace recovery
under arbitrary predictor or dose distributions. Tied selected eigenvalues can
give different bases across numerical libraries.
The finite-sample dose kernel uses the individual projected coordinates, so
rotating a basis or rescaling predictors can change the fitted rule. Apply
any chosen predictor preprocessing consistently to training and new data.

`ortho_optim` minimizes a scalar Python objective under `B.T @ B == I`.
Set `maximize=True` to maximize it. The optional analytic gradient must have
the same shape as `B`. Additional positional and keyword arguments pass to
both callbacks. Without `grad`, the solver uses forward differences.
`fn_Seq[:itr]` contains completed iterations; the remaining allocated entries
are zero. Inputs and callback results must be finite.

The hMAVE `cv` score uses squared prediction errors for Gaussian event-time
kernel values among observed events. It is not a calibrated survival-probability
or hazard prediction loss.

`silverman(d, n)` gives the bandwidth above. `dist_cross(x1, x2)` and
`kernel_weight(x1, x2)` return `exp(-squared_distance)` between rows. The
compatibility `kernel` and `dist` arguments of `kernel_weight` are unused,
matching R. `distance(s1, s2, type="dist", x=None)` compares subspaces:
`dist` is projector Frobenius distance, `trace` and `canonical` are
similarities, and `sine` retains R's square root of the sum of fourth powers
of principal-angle sines. `canonical` requires predictor data `x`.

## Plotting and example data

`view_dr_surv(x, y, censor, B, type="2D", FUN="log", ..., ax=None)` plots
observed times against one reduced direction. Events and censored observations
use distinct markers. `type="3D"` displays a kernel-weighted Kaplan-Meier
surface. `B` must be a vector or a one-column matrix. The surface bandwidth is
on the projected coordinate scale, defaulting to
`silverman(1, n) * std(x @ B, ddof=1)`.

The default logarithm requires positive times. Use `FUN="identity"` for
untransformed time, or supply an order-preserving transformation. `xlab`,
`ylab`, and `zlab` set axis labels; use an appropriate `ylab` when transforming
time. `legend_add=False` hides the 2D legend. The function lazily imports
Matplotlib, returns the axes, and never opens a display. Save with
`ax.figure.savefig("survival.png")`, or pass existing Matplotlib axes for an
interactive application. Its plotting backend replaces R's base/rgl graphics;
it does not reproduce their styling.

`load_skcm()` returns the R package's bundled TCGA melanoma example data as a
dictionary. `clinical` has 469 rows and the four columns in
`clinical_columns`: `Time`, `Censor`, `Gender`, and `Age`. There are 156 observed
events and eight missing ages. `genes` contains the matching 469 rows and 20
gene columns in `gene_names`. `subject_ids` identifies the rows; their order
agrees across the two arrays. Missing ages remain `NaN`, and the gene values
retain their source scaling. Handle missingness explicitly before fitting.
Each load returns independent arrays. The bundled provenance records the R
source revision and file checksums; neither pandas nor R is required to load
the data.

## References

See [references and citation](references.md) for software attribution and the
method papers. The statistical references and their roles follow the published
R package:

- Ma and Zhu (2012), semiparametric dimension reduction,
  [JASA](https://doi.org/10.1080/01621459.2011.646925).
- Ma and Zhu (2013), efficient estimation in sufficient dimension reduction,
  [Annals of Statistics](https://doi.org/10.1214/12-AOS1072).
- Sun, Zhu, Wang, and Zeng (2019), counting process methods for censored outcomes,
  [Biometrika](https://doi.org/10.1093/biomet/asy064).
- Zhou, Zhu, and Zeng (2021), personalized dose finding,
  [Biometrika](https://doi.org/10.1093/biomet/asaa087).
- Wen and Yin (2013), feasible optimization with orthogonality constraints,
  [Mathematical Programming](https://doi.org/10.1007/s10107-012-0584-1).
