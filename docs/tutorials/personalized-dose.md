# Personalized doses

Learn a continuous dose rule from predictors, observed doses, and rewards,
with larger rewards preferred. Direct learning optimizes an estimated reward
criterion; pseudo-direct learning first fits the conditional reward surface,
then constructs a dose rule.

## Choose the target and study interpretation

Supply `X` with shape `(n, p)` and observed doses `a` and rewards `r` with
shape `(n,)`. Doses must vary. Both methods return directions and continuous
recommendations in the original dose units. They need not estimate the same
dimension-reduction subspace because the reward surface can depend on
predictors that do not change the preferred dose.

A causal interpretation requires consistency (the observed reward equals
the potential reward at the observed dose), no unmeasured dose-outcome
confounding given `X`, and adequate dose support in the relevant reduced
space. These are study assumptions; the function does not check them. See
[Zhou, Zhu, and Zeng (2021)](../references.md#personalized-dose-learning)
for the model and identification conditions.

## Fit a direct dose rule

In this simulation the optimal dose is `2 + 0.8 * X[:, 0]`. Mean reward
decreases as the observed dose moves away from that value. Fit with an
intercept, then compare recommendations with known optimal doses at 50 new
predictor values.

```python
import numpy as np
from orthodr import orthoDr_pdose

rng = np.random.default_rng(34)
X = rng.uniform(-1, 1, size=(100, 3))
a = rng.uniform(0, 4, size=len(X))
best_dose = 2 + 0.8 * X[:, 0]
r = 3 * np.exp(-((a - best_dose) / 0.7) ** 2) + rng.normal(0, 0.1, len(X))

fit = orthoDr_pdose(
    X, a, r, ndr=1, method="direct", intercept=True,
    lambda_reward=0.1, keep_data=True, maxitr=100,
)
new_X = rng.uniform(-1, 1, size=(50, 3))
recommended = fit.predict(new_X)
truth = 2 + 0.8 * new_X[:, 0]
print("Converged:", fit.converge, "Stop reason:", fit.stop_reason)
print(f"Direction residual: {fit.gradient_norm['B']:.2e}")
print(f"Policy residual: {fit.gradient_norm['policy']:.2e}")
print(f"Dose RMSE: {np.sqrt(np.mean((recommended - truth) ** 2)):.6f}")
```

Observed output from a Windows source build:

```text
Converged: True Stop reason: gradient_tolerance
Direction residual: 3.54e-07
Policy residual: 8.39e-13
Dose RMSE: 0.081609
```

The dose RMSE is about 0.082 in the simulated dose units. This evaluates the
50 new recommendations against a known rule, separately from the numerical
residuals. Neither this result nor convergence calibrates `lambda_reward=0.1`
for another population, reward scale, or dose unit.

### Understand the direct objective

Without an intercept, direct learning minimizes

$$
Q(B,w)=-\widehat V(B,w)+\lambda_{\mathrm{reward}}w^\top w,
\qquad B^\top B=I.
$$

`fit.value` is the estimated training mean reward; `fit.penalty` is the
coefficient penalty; and `fit.fn` is `-fit.value + fit.penalty`. The penalty
has no additional division by sample size and is a squared coefficient norm,
not a squared RKHS norm. Its strength depends on reward units, dose units, bandwidth,
and training centers. Comparing `fit.fn` across different penalties does not
select a penalty for decisions on new observations.

With an intercept, the estimated reward also depends on $\alpha$, which is
unpenalized and constrained by `intercept_bounds`.

The default `intercept=False` fits the uncentered rule
$d(x)=k_B(x)^\top w$. A positive coefficient penalty shrinks that rule toward
zero dose, so the dose origin matters. The example opts into
`intercept=True`, which fits

$$
d(x)=\alpha+\{k_B(x)-\mu_B\}^{\top}w,
\qquad \mu_B=\frac{1}{n}\sum_{i=1}^n k_B(x_i).
$$

The intercept is unpenalized and equals the average fitted training dose.
`intercept_bounds` bounds that average, using the observed dose range by
default. It does **not** bound individual recommendations. The intercept
option changes the model and does not guarantee better decisions.

## Fit the reward surface first

Pseudo-direct learning uses `method="pseudo_direct"`. This separate simulation
has an optimal dose depending on one index of four predictors. A vector of
`lambda_init` values requests generalized cross-validation (GCV) for the final
ridge dose rule; it does not tune the reward-surface bandwidth or select a
causal evaluation criterion.

```python
pseudo_rng = np.random.default_rng(1204)
pseudo_X = pseudo_rng.normal(size=(120, 4))
pseudo_a = pseudo_rng.uniform(0, 4, len(pseudo_X))
pseudo_optimal = 2 + 0.8 * np.tanh(
    (pseudo_X[:, 0] + pseudo_X[:, 1]) / np.sqrt(2)
)
pseudo_r = 3 * np.exp(-((pseudo_a - pseudo_optimal) / 0.7) ** 2)
pseudo_r += pseudo_rng.normal(0, 0.1, len(pseudo_X))

pseudo_fit = orthoDr_pdose(
    pseudo_X, pseudo_a, pseudo_r, method="pseudo_direct", ndr=1,
    lambda_init=[0.05, 0.1, 0.3], intercept=True, K=12,
    keep_data=True, ncore=1,
)
pseudo_new_X = pseudo_rng.normal(size=(1000, 4))
pseudo_truth = 2 + 0.8 * np.tanh(
    (pseudo_new_X[:, 0] + pseudo_new_X[:, 1]) / np.sqrt(2)
)
pseudo_recommended = pseudo_fit.predict(pseudo_new_X)
pseudo_dose_rmse = np.sqrt(np.mean((pseudo_recommended - pseudo_truth) ** 2))
pseudo_reward_loss = np.mean(
    3 - 3 * np.exp(-((pseudo_recommended - pseudo_truth) / 0.7) ** 2)
)
print("Converged:", pseudo_fit.converge, "Stop reason:", pseudo_fit.stop_reason)
print(f"Normalized gradient norm: {pseudo_fit.gradient_norm:.2e}")
print(f"Reward fitting RMSE: {pseudo_fit.fn:.6f}")
print("Selected ridge penalty:", pseudo_fit.lambda_init)
print(f"Dose RMSE: {pseudo_dose_rmse:.6f}")
print(f"Mean reward loss: {pseudo_reward_loss:.6f}")
```

Observed output from a Windows source build:

```text
Converged: True Stop reason: gradient_tolerance
Normalized gradient norm: 1.01e-07
Reward fitting RMSE: 0.413077
Selected ridge penalty: 0.05
Dose RMSE: 0.110015
Mean reward loss: 0.071130
```

The reward fitting RMSE measures training reward-surface fit in reward units.
The dose RMSE instead compares 1,000 new recommendations with known optimal
doses. Mean reward loss is the average reduction from the optimal expected
reward of 3, using the known simulated reward function. These quantities
measure different properties. The direct and pseudo-direct examples use
different simulations, so their numbers do not compare the methods.

Pseudo-direct optimization uses mean squared residuals after centering and
scaling rewards by their sample standard deviation. The returned `fn` remains
raw reward RMSE. `gradient_norm` is the normalized objective's constrained
gradient norm. The training smoother includes each observation itself, so a
small training RMSE at a narrow bandwidth does not establish accurate dose
decisions.

The final pseudo-direct ridge rule uses a squared RKHS norm penalty. Its optional
`intercept=True` adds an unbounded, unpenalized constant to that rule, with
intercept-aware GCV. This differs from direct learning's bounded average-dose
intercept. Both methods default to no intercept.

## Keep the penalty and grid roles separate

| Argument | Direct learning | Pseudo-direct learning |
| --- | --- | --- |
| `lambda_init`, default `0.1` | Ridge initialization only; a vector requests initial GCV | Ridge penalty for the final dose rule; a vector requests GCV |
| `lambda_reward`, nominal default `0.1` | Squared coefficient penalty in reward optimization | Omit this argument |
| `K`, default `floor(sqrt(n))` | Number of grid doses for initialization | Number of grid doses for constructing final ridge targets |
| `intercept_bounds` | Bounds the mean training recommendation when an intercept is enabled | Omit this argument |

The grid spans the observed dose range. Both final rules give continuous
predictions and do not clip them to that range. Inspect support for proposed
recommendations and assess penalties and bandwidths through evaluation suited
to the data and study design. The nominal defaults are not a calibration study.

Both methods use pSAVE when `B_initial=None`. pSAVE is one available starting
basis, not a guarantee of accurate directions or recommendations. A supplied
full-column-rank `(p, ndr)` basis, including a random basis, bypasses pSAVE
and is orthonormalized before fitting. This also permits a start when dose
groups lack the rank or reward-slice information required by pSAVE. Different
starts can lead to different results.

## Read stopping diagnostics

For both dose methods, only `stop_reason="gradient_tolerance"` sets
`converge=True`. An iteration limit, stagnation, failed line search, or
numerical failure does not certify convergence. Direct learning checks both
direction and policy residuals; pseudo-direct learning checks the normalized
constrained gradient for its reward-surface objective.

Numerical stationarity does not guarantee a global optimum or good decisions
on new observations. The [API reference](../api.md#personalized-dose-learning)
documents exact diagnostics, scaling identities, bandwidth conventions, and
intercept restrictions. The standalone
[pseudo-direct example](https://github.com/teazrq/orthoDr-Python/blob/main/examples/pseudo_direct.py)
contains the second workflow as an executable script.
