# Python and R

Use corresponding R and Python methods with the same data conventions, while
checking the numerical scope of a comparison. The Python package runs fitting
and prediction without an R session; R is needed only to regenerate the R
reference fixtures.

## Match the interfaces

| R | Python | Meaning |
| --- | --- | --- |
| `orthoDr_reg` | `orthoDr_reg` | Regression directions and mean prediction |
| `orthoDr_surv` | `orthoDr_surv` | Censored-outcome directions and survival prediction |
| `orthoDr_pdose` | `orthoDr_pdose` | Direct or pseudo-direct dose learning |
| `B.initial` | `B_initial` | Starting direction matrix |
| `keep.data` | `keep_data` | Retain training state for prediction |
| `lambda` | `lambda_init` | Initialization ridge penalty, or final ridge penalty for pseudo-direct learning |
| `lambda.reward` | `lambda_reward` | Direct-learning coefficient penalty |
| `intercept.bounds` | `intercept_bounds` | Direct-learning mean-dose bounds |

Both interfaces use observations in rows. In survival fitting, `censor=1`
means an observed event and `censor=0` means right censoring. Python's
`FitResult` supports dictionary keys and attribute access, with underscores
standing for dots in diagnostic attributes.

Python's `fit.predict(new_X)` returns a one-dimensional array for regression
and doses, and a dictionary with `surv` and `timepoints` for survival.
The top-level Python `predict(fit, new_X)` always returns a dictionary.
An R serialized fit cannot be loaded as a Python pickle.

## Understand the numerical comparison

The port retains the pinned R source's native numerical routines. Python
implements the surrounding preprocessing, initializers, prediction, plotting,
and data access. Saved R fixtures test regression, survival, both dose methods,
initializers, the optimizer, and prediction using recorded inputs and fitted
states. Fixed-state objective, derivative, and prediction checks separate
formula agreement from an optimizer's trajectory.

Nonconvex updates can amplify floating-point differences. A tightly matching
early trajectory need not give identical later unfinished iterates, and a
finite-difference solver can be sensitive to very small changes in its
starting basis. Tied grid scores or GCV scores can also select different
discrete winners across numerical libraries. These observations do not
establish uniform numerical agreement or statistical equivalence.

Compare objective definitions, preprocessing, starting state, and stopping
diagnostics before comparing raw direction coefficients. Direction signs
alone can differ without changing the subspace. Convergence, prediction
agreement, and statistical performance are different checks.

## Source and build coverage

| Evidence | Location |
| --- | --- |
| Pinned numerical source | [R source revision](https://github.com/teazrq/orthoDr/commit/a9588d8dd73c39188d1c46f371763addb10bee9a) |
| Test scope, observed errors, and build environment | [Validation report](https://github.com/teazrq/orthoDr-Python/blob/main/VALIDATION.md) |
| Reference-source setup and fixture regeneration | [Fixture guide](https://github.com/teazrq/orthoDr-Python/blob/main/tests/fixtures/README.md) |
| Installed-package Linux checks | [CI workflow and completed runs](https://github.com/teazrq/orthoDr-Python/actions/workflows/ci.yml) |

The development revision includes coordinated pseudo-direct and pSAVE
corrections. Pseudo-direct convergence requires a small normalized constrained
gradient; its returned `fn` remains raw reward RMSE. pSAVE pools group SAVE
information in common coordinates and returns directions in the supplied
predictor units. This correction can change automatic dose starts; it does
not establish better dose decisions.

Local build validation covers Windows with shared Rtools and Armadillo
headers. CI targets Ubuntu with Python 3.10 and 3.14. macOS builds have not
been validated. See the [installation guide](../tutorials/get-started.md)
for the corresponding prerequisites.

The [R documentation](https://teazrq.github.io/orthoDr/) describes the R
interface. The [Python API](../api.md) documents the behavior of this source
version, including the distinct dose objectives and intercept options.
