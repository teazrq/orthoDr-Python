# orthoDr for Python

Estimate a small set of linear combinations of continuous predictors for
regression, right-censored survival outcomes, or personalized dose learning.
`orthodr` ports the statistical methods of the
[orthoDr R package](https://teazrq.github.io/orthoDr/) to NumPy. Fitting and
prediction run without an R session.

Version **2.1.0** is available from
[GitHub](https://github.com/teazrq/orthoDr-Python). It uses the R package's C++
numerical routines and implements preprocessing, initializers, and prediction
in Python. Install it from source using the instructions below. See
[validation](VALIDATION.md) for the R comparison and its numerical limits.

See the [Python documentation](https://teazrq.github.io/orthoDr-Python/) for
installation, task-oriented tutorials, and the API reference.

## Installation

Use an existing shared Python installation with NumPy, SciPy, setuptools,
wheel, and pybind11. Python 3.10 or later is required by the source. Install
missing Python dependencies in the shared user or machine installation:

```bash
python -m pip install --user "numpy>=1.24" "scipy>=1.10" "setuptools>=77" wheel "pybind11>=3.0"
```

Clone the source, build the native extension for checkout examples and tests,
then install the package into the shared user installation:

```bash
git clone https://github.com/teazrq/orthoDr-Python.git
cd orthoDr-Python
python setup.py build_ext --inplace
python -m pip install --user --no-build-isolation --no-deps .
```

The source build also needs a C++17 compiler, Armadillo headers, BLAS, and
LAPACK. On Windows, the build uses an existing shared Rtools 4.5 installation.
It can use the standalone Armadillo headers shipped with a shared
RcppArmadillo installation; these are build inputs, and the resulting Python
extension does not link to R. Set `ORTHODR_RTOOLS` and
`ORTHODR_ARMADILLO_INCLUDE` to select existing installations if needed. The
default Windows paths are documented in `tools/build_native.py`.

On Debian or Ubuntu, the system dependencies are `g++`, `libarmadillo-dev`,
`liblapack-dev`, and `libblas-dev`. The setuptools build enables OpenMP on Linux.
The validation report lists the platforms actually tested. macOS builds have
not been validated.

Plotting additionally uses Matplotlib. Running the test suite uses pytest.
Neither is required for fitting or prediction.

## First regression analysis

Rows are observations and columns are predictors. Supply finite numeric arrays
and choose `ndr`, the number of directions to estimate. This example holds out
new observations before fitting and compares predictions with their responses.

```python
import numpy as np
from orthodr import orthoDr_reg

rng = np.random.default_rng(21)
X = rng.normal(size=(160, 4))
y = 2 * X[:, 0] - X[:, 1] + rng.normal(scale=0.5, size=160)

fit = orthoDr_reg(X[:120], y[:120], ndr=1, method="sir", keep_data=True)
prediction = fit.predict(X[120:])
print("Converged:", fit.converge, "Iterations:", fit.itr)
print("Held-out MSE:", np.mean((prediction - y[120:]) ** 2))
print("Reduced coordinates:", fit.transform(X[120:]).shape)
```

`fit.B` has shape `(number_of_predictors, ndr)` and `fit.transform(X)` returns
`X @ fit.B`. Regression predictions are kernel estimates of the mean response.
The basis describes a subspace: reversing a direction's sign does not change
that subspace. An iteration limit can return a usable last estimate, but it
does not establish convergence.

Methods are `"sir"`, `"save"`, `"phd"`, `"local"`, and `"seff"`. With one
varying predictor, the default starting basis is `[[1]]` for every method.
With multiple predictors, automatic starting directions use SIR, SAVE, and
PHD. The one-direction LOCAL method fits three starts and selects by training
kernel R-squared. That selection score is not an estimate of performance on
new data.

The regression methods use one pooled predictor scale for optimization. PHD
uses centered outer products. SAVE retains its supplied-unit estimating
equation. Supply new observations in the same coordinates used for training.

Regression requires at least two observations and a varying response; a scalar
predictor must also vary. Automatic starts with multiple predictors require
full centered predictor rank and at least two observations in every SAVE
response slice, including when fitting another method. Small samples may not
support these slices. Supply a full-column-rank `B_initial` with shape
`(X.shape[1], ndr)` to bypass the sliced initializers; training projections must
still have positive variance.

## Survival outcomes

`y` contains observed times, and `censor=1` means an observed event;
`censor=0` means right censoring. Despite its name, this argument is an event
indicator. The example below generates event and censoring times separately.

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
print("First event times:", curves["timepoints"][:3])
```

`curves["surv"][j, i]` estimates survival beyond `curves["timepoints"][j]`
for new observation `i`. Rows follow sorted training event times, including
repeated times. Tied events use the complete risk set at that time. Predictions
use a kernel-weighted Kaplan-Meier estimator.

Methods are `"forward"` (one direction), `"dn"`, and `"dm"`. Survival fitting
standardizes predictors internally and returns directions on the original
predictor scale. Their columns have unit length but, after rescaling, need not
be mutually orthogonal.

## Personalized doses

Provide continuous observed doses `a` and rewards `r`, with larger rewards
preferred. Direct learning estimates a dose rule. Pseudo-direct learning first
estimates a reward surface, so the two methods need not target the same
dimension-reduction subspace.

A causal interpretation of a learned dose rule requires consistency (the
observed reward equals the potential reward at the observed dose), no
unmeasured dose-outcome confounding given `X`, and adequate dose support in the
relevant reduced space. These are study assumptions; the function does not
check them. See the
[dose-learning paper](https://doi.org/10.1093/biomet/asaa087) for the model and
identification conditions.

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
print("Final residuals:", fit.gradient_norm)
print("Dose RMSE against simulation truth:", np.sqrt(np.mean((recommended - truth) ** 2)))
```

This is a reproducible use example, not a calibration study. The penalty `0.1`
is the R default, a nominal starting value. Inspect the reported convergence
and residuals, and assess penalty and bandwidth using policy evaluation suited
to the data and study design. Numerical convergence does not guarantee a
global optimum or good decisions on new observations.

Two penalty arguments serve different purposes:

| Argument | Direct learning | Pseudo-direct learning |
| --- | --- | --- |
| `lambda_init` | Ridge initialization only; a vector invokes initial GCV | Ridge penalty for the final dose rule |
| `lambda_reward` | Squared coefficient penalty during reward optimization | Omit this argument |

Direct learning minimizes

$$
Q(B,w)=-\widehat V(B,w)+\lambda_{\mathrm{reward}}\,w^\top w,
\qquad B^\top B=I.
$$

Here `fit.value` is the estimated training mean reward, and
`fit.fn == -fit.value + fit.penalty`. The coefficient penalty has no extra
division by sample size. Its strength depends on the reward and dose units,
bandwidth, and training centers. Comparing `fit.fn` across different penalties
does not select a penalty for predictive performance.

For direct learning, the default `intercept=False` uses an uncentered kernel rule:

$$
d(x)=k_B(x)^\top w.
$$

Here `k_B(x)` contains Gaussian kernel comparisons with the projected training
observations. A positive coefficient penalty shrinks this rule toward zero
dose, so its effect depends on the dose origin. With `intercept=True`, the
fitted rule is

$$
d(x)=\alpha+\{k_B(x)-\mu_B\}^{\top}w,
\qquad \mu_B=\frac{1}{n}\sum_{i=1}^{n}k_B(x_i).
$$

The unpenalized intercept is the average fitted training dose, and the
coefficients describe variation around that average.
`intercept_bounds` bounds that average, using the observed dose range by
default. Individual recommendations are continuous and can fall outside those
bounds or the observed range. The intercept option changes the model and does
not guarantee better decisions. See the
[dose reference](docs/api.md#personalized-dose-learning) for the exact
convergence fields and scaling conventions.

Pseudo-direct learning uses `method="pseudo_direct"` to fit the reward surface
before constructing a ridge dose rule. When several grid doses are tied for
the highest estimated reward within numerical precision, the lowest becomes
the training target. The fit reports `gradient_norm` and
`stop_reason` for a reward-standardized mean squared error objective; `fn`
remains raw reward RMSE. Only a small constrained gradient reports convergence.
Its optional `intercept=True` adds an unbounded, unpenalized constant to the
final ridge rule, with intercept-aware GCV. The default remains `False`.
See the runnable [pseudo-direct example](examples/pseudo_direct.py) and the
[API explanation](docs/api.md#personalized-dose-learning) for its distinct
penalty, bandwidth, and stopping diagnostics.

## Other operations

`CP_SIR`, `pSAVE`, and `hMave` expose the initial dimension-reduction estimators.
`ortho_optim` optimizes a Python objective under orthonormal-column constraints,
with an optional analytic gradient. `silverman`, `distance`, `dist_cross`, and
`kernel_weight` provide the R package's numerical utilities. See the
[API guide](docs/api.md) for arguments, output shapes, and plotting/data access.
`view_dr_surv` provides Matplotlib survival views, and `load_skcm` loads the
bundled clinical and gene-expression example data with aligned subject IDs.

Retain training data with `keep_data=True` when predictions are needed.
`FitResult` is a dictionary with attribute access: `fit.B` and `fit["B"]`
refer to the same array, and `fit.gradient_norm` accesses `fit["gradient.norm"]`.
Save and load a fit with Python's `pickle` to retain its training projection
scales and intercept state, including kernel means for direct intercept rules. Load pickle
files only from trusted sources.

The functional API follows R names while using Python argument spelling:
`B_initial`, `keep_data`, `lambda_init`, `lambda_reward`, and
`intercept_bounds`. The default `ncore=1` uses one thread; `ncore=0` requests
automatic detection for native routines that support parallel computation.
Both analytic dose solvers run serially.

## Source and validation

The native source uses R source commit
[`a9588d8`](https://github.com/teazrq/orthoDr/commit/a9588d8dd73c39188d1c46f371763addb10bee9a) as its baseline, based on orthoDr commit
[`d99bfa8`](https://github.com/teazrq/orthoDr/commit/d99bfa869ab211f974a7a045e8fa6ede23639a48).
The current build adds a fingerprinted update from the R development source
for regression, survival optimization, and dose fitting and prediction.
`tools/vendor_core.py` checks the source
and records the R and generated Python hashes in `src/cpp/provenance.json`.
The R-derived test fixtures include
their source revision, input arrays, fitted
states, and predictions. The [fixture guide](tests/fixtures/README.md) explains
how to obtain the reference source and regenerate them with the R development
dependencies. R is not needed to run the Python tests against saved fixtures.

The default example check uses this checkout; `--installed` checks the package
in the shared installation.

```bash
python -m pytest -q
python tools/verify_examples.py
python tools/verify_examples.py --installed
```

The original authors and third-party dependencies are acknowledged in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This combined distribution
uses GPL version 3 or later; see [LICENSE](LICENSE).
