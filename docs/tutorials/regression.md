# Regression

Fit a low-dimensional representation of continuous predictors and use kernel
smoothing in that representation to predict the mean response. This guide
fits one direction, evaluates held-out prediction error, and explains the
method and initialization choices.

## Fit one direction and predict

Here the response depends on the index $2X_1-X_2$ plus independent noise.
The data-generating mechanism therefore gives a known direction for this
simulation. Reserve 40 observations before fitting the other 120.

```python
import numpy as np
from orthodr import orthoDr_reg

rng = np.random.default_rng(21)
X = rng.normal(size=(160, 4))
y = 2 * X[:, 0] - X[:, 1] + rng.normal(scale=0.5, size=160)

fit = orthoDr_reg(X[:120], y[:120], ndr=1, method="sir", keep_data=True)
prediction = fit.predict(X[120:])
print("Converged:", fit.converge, "Iterations:", fit.itr)
print(f"Held-out MSE: {np.mean((prediction - y[120:]) ** 2):.6f}")
print("Basis shape:", fit.B.shape)
print("Reduced coordinates:", fit.transform(X[120:]).shape)
```

Observed output from a Windows source build:

```text
Converged: True Iterations: 10
Held-out MSE: 0.390764
Basis shape: (4, 1)
Reduced coordinates: (40, 1)
```

`fit.B` maps four predictors to one coordinate. The predictions are
kernel-weighted averages of training responses, with larger weights for
nearby reduced coordinates. The held-out MSE is about 0.391 in squared
response units. It assesses prediction on these 40 simulated observations,
not recovery of the direction itself.

## Interpret the directions

For a predictor column vector $x$, the reduced representation is

$$
z=B^\top x.
$$

For an array of rows, the same calculation is `X @ fit.B`. A direction and
its negative describe the same one-dimensional subspace, so compare subspaces
rather than raw coefficient signs. The `distance` utility provides such
comparisons; its `type="dist"` option uses the Frobenius distance between
projection matrices.

The function does not choose the dimension `ndr`. Use scientific knowledge
and evaluation on separate data to assess candidate dimensions and bandwidths.
The default bandwidth is a sample-size rule, described in the
[API reference](../api.md#common-fitting-arguments), rather than a
cross-validated choice. Training fit alone does not measure predictive
performance on new observations.

## Choose the fitting method

| `method` | Fitting criterion |
| --- | --- |
| `"sir"` | Semiparametric SIR estimating equation, the default |
| `"save"` | Semiparametric SAVE estimating equation |
| `"phd"` | Semiparametric PHD estimating equation |
| `"local"` | Local conditional-mean estimation |
| `"seff"` | Efficient estimating equation |

The SIR, SAVE, and PHD fitting methods optimize their estimating-equation
objectives; they are distinct from the sliced estimators used as starting
values. See [Ma and Zhu's regression papers](../references.md#regression)
for the semiparametric construction and efficient estimation.

SIR, PHD, LOCAL, and SEFF use one common predictor scale so that changing the
units of all predictors together preserves their optimization objectives. PHD
uses centered outer products. SAVE adjusts the scale of its objective for
stopping and retains its supplied-unit equation, so changing units can still
change its fitted directions. Supply new observations in the same coordinates
used for training, including when you supply a starting basis.

With one varying predictor and no supplied start, every method uses the scalar
basis `[[1]]`. At least two observations and a varying response are required.
The usual objective and solver still run; zero optimizer iterations are
expected because the scalar basis has no continuous direction to optimize.

With multiple predictors, automatic initialization computes SIR, SAVE, and
residual-PHD candidates. Most configurations select the candidate with the
smallest initial objective; two-direction fits also consider the leading SIR
and SAVE directions jointly. One-direction LOCAL instead fits three starts
and selects by training kernel R-squared. That score measures training fit,
not held-out prediction error.

## Supply a starting basis when needed

For multiple predictors, automatic sliced initialization requires full centered
predictor rank and at least two observations in every SAVE response slice.
This requirement also applies to other regression methods because all three
initializers are computed. A small sample or collinear predictors can prevent
automatic initialization.

Supply a full-column-rank matrix `B_initial` of shape `(p, ndr)` to bypass the
sliced initializers. Its columns are orthonormalized before fitting, and
training projections must still have positive variance. For the simulated
one-direction example above, a coordinate direction is a valid alternative
starting point:

```python
B_initial = np.array([[1.0], [0.0], [0.0], [0.0]])
fit_from_start = orthoDr_reg(
    X[:120], y[:120], ndr=1, method="sir", B_initial=B_initial,
    keep_data=True,
)
prediction_from_start = fit_from_start.predict(X[120:])
```

A different start can lead to a different fitted solution. Neither the
automatic start nor this coordinate start guarantees a global optimum.

## Read convergence and preserve prediction state

Inspect `fit.converge` and `fit.itr`. Regression retains the R solver's stopping
criteria, which can report convergence after small parameter and objective
changes. Reaching an iteration limit returns the last estimate and does not
establish convergence. Numerical stopping and predictive accuracy are
separate questions.

Prediction requires the training data retained by `keep_data=True`. Saving
the fit with Python's `pickle` preserves its prediction state. Load pickle
files only from trusted sources. Use the same predictor order and external
preprocessing at prediction as at fitting.

See the [regression API](../api.md#regression) for all arguments, or continue
to [survival outcomes](survival.md) and [personalized doses](personalized-dose.md).
