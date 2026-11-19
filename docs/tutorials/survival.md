# Survival outcomes

Fit directions for a right-censored event time and predict conditional survival
curves for new observations. The returned curves use kernel-weighted
Kaplan-Meier estimates in the fitted reduced coordinates.

## Represent observed times and events

Supply the observed time `y = min(event_time, censor_time)` and an event
indicator through the argument named `censor`. Despite that argument's name,
**1 means an observed event and 0 means right censoring**.

The simulation below generates event times whose scale depends on the first
predictor, and independent censoring times. Times are in arbitrary simulation
units. Fit the martingale method on 110 observations and predict the 30 held
out before fitting.

```python
import numpy as np
from orthodr import orthoDr_surv

rng = np.random.default_rng(12)
X = rng.normal(size=(140, 4))
event_time = rng.exponential(scale=np.exp(-0.7 * X[:, 0]))
censor_time = rng.exponential(scale=2.0, size=len(X))
time = np.minimum(event_time, censor_time)
event = (event_time <= censor_time).astype(int)

fit = orthoDr_surv(
    X[:110], time[:110], event[:110], ndr=1, method="dm", keep_data=True
)
curves = fit.predict(X[110:])
print("Converged:", fit.converge, "Iterations:", fit.itr)
print("Survival matrix:", curves["surv"].shape)
print("First event times:", np.round(curves["timepoints"][:3], 6))
```

Observed output from a Windows source build:

```text
Converged: True Iterations: 11
Survival matrix: (69, 30)
First event times: [0.009067 0.012287 0.018694]
```

## Read the survival matrix

`curves["surv"][j, i]` estimates survival beyond
`curves["timepoints"][j]` for held-out observation `i`. There are 69 observed
training events in this example, so the result has 69 rows and 30 columns.
The time grid contains sorted training event times, including repeated
times. Tied events use the complete risk set at that time.

A curve represents an estimate of

$$
S(t\mid z)=\Pr(T>t\mid B^\top X=z),
$$

where $T$ is the event time and $z$ is the reduced predictor value.
The survival methods assume event and censoring times are independent
conditional on the covariates `X`. Prediction also relies on an adequate
fitted reduction and local data support; the function does not check these
study conditions. The methods and their statistical conditions are described by
[Sun, Zhu, Wang, and Zeng (2019)](../references.md#survival).

The matrix shape confirms the prediction layout. It does not assess curve
calibration or survival prediction error. Those require an evaluation suited
to the censoring mechanism and the scientific prediction horizon.

## Choose a method and dimension

| `method` | Number of directions when omitted | Role |
| --- | --- | --- |
| `"forward"` | 1 | Forward estimating equation; always uses one direction |
| `"dn"` | 2 | Counting-process estimating equation |
| `"dm"` | 2 | Martingale estimating equation, the default |

`ndr` is capped at the number of predictors. Each predictor must have positive
sample variance, and at least one event is required. Automatic initialization
uses counting-process SIR.

Survival fitting standardizes predictors internally. A supplied `B_initial`
uses those standardized predictor coordinates. The returned `fit.B` is mapped
back to the original predictor scale, so `fit.transform(new_X)` remains
`new_X @ fit.B`. Its columns have unit length but need not remain mutually
orthogonal after that rescaling.

Inspect `fit.converge` and `fit.itr`. The inherited survival solver can stop on
small changes; its convergence flag does not establish a global optimum or
accurate survival probabilities on new observations.

## Plot and inspect example data

The optional `view_dr_surv` utility plots observed time against one reduced
direction, distinguishing events and censored observations. Its `type="3D"`
view displays a weighted survival surface. It returns Matplotlib axes so you
can label and save a figure. The default time transformation is logarithmic,
which requires positive observed times.

```python
from orthodr import view_dr_surv

ax = view_dr_surv(
    X[:110], time[:110], event[:110], B=fit.B,
    FUN="log", type="3D", ylab="Log time",
)
```

The time knots combine every observed event time with the 0th through 100th
pooled follow-up percentiles, selecting observed values without interpolation
and removing duplicates. They span the actual follow-up range on the chosen
time scale. Survival is calculated on the original follow-up times. Knots
that coincide on the display show the latest knot's survival probability.
A single distinct time is shown as a line. Use `FUN="identity"`
and `ylab="Time"` to keep the original time units. See
[plotting and data access](../api.md#plotting-and-example-data) for the full
options and the optional shared Matplotlib dependency.

`load_skcm()` loads aligned clinical and gene-expression arrays from the
bundled melanoma example. It preserves missing ages as `NaN`, so handle those
before fitting. Subject IDs identify the same row order in both arrays.

See the [survival API](../api.md#survival) for defaults and prediction details,
or the [Python and R comparison](../reference/r-python.md) for numerical scope.
