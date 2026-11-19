# Get started

Install `orthodr`, fit one regression direction, and evaluate predictions on
observations held out before fitting. This page uses version **2.1.0** with a
shared user-wide or machine-wide Python installation.

## Install from source

Use Python 3.10 or later and Git. The native extension also requires a C++17
compiler, Armadillo headers, BLAS, and LAPACK. Install those shared system
dependencies for your platform before building the package.

### Debian or Ubuntu

Install the compiler and numerical libraries system-wide:

```bash
sudo apt-get update
sudo apt-get install -y g++ libarmadillo-dev liblapack-dev libblas-dev
```

The setuptools build enables OpenMP on Linux. The
[CI workflow](https://github.com/teazrq/orthoDr-Python/actions/workflows/ci.yml)
tests source installation, the installed package, and runnable examples on
Ubuntu with Python 3.10 and 3.14.

### Windows

Use a shared Rtools 4.5 installation for its compiler, BLAS, and LAPACK.
Armadillo headers can come from a shared standalone installation or the
`include` directory of a shared RcppArmadillo installation. These are build
inputs; the resulting extension does not link to R.

The build searches these default locations:

| Input | Default location |
| --- | --- |
| Rtools toolchain | `C:/rtools45/x86_64-w64-mingw32.static.posix` |
| Armadillo headers | `C:/Program Files/R/R-4.6.1/library/RcppArmadillo/include`, then the Rtools toolchain's `include` directory |

For other existing shared locations, set `ORTHODR_RTOOLS` to the toolchain
directory and `ORTHODR_ARMADILLO_INCLUDE` to the directory containing the file
`armadillo`. The [Windows build helper](https://github.com/teazrq/orthoDr-Python/blob/main/tools/build_native.py)
defines the search paths. macOS builds have not been validated.

### Build and install the Python package

Run these commands in a terminal using your shared Python installation. The
dependency command installs missing Python packages in the shared user
installation; it does not create a project environment.

```bash
git clone https://github.com/teazrq/orthoDr-Python.git
cd orthoDr-Python
python -m pip install --user "numpy>=1.24" "scipy>=1.10" "setuptools>=77" wheel "pybind11>=3.0"
python setup.py build_ext --inplace
python -m pip install --user --no-build-isolation --no-deps .
```

The in-place build supports examples run from the checkout. The final command
installs the package for use from other directories. If your Python is managed
system-wide and disables `--user` installs, install these dependencies and the
package through that shared installation's supported package-management route.

### Check which package Python loads

Run this from outside the cloned directory to inspect the installed package:

```python
import orthodr

print(orthodr.__version__)
print(orthodr.__file__)
```

The version should read `2.1.0`. The path identifies the installation Python loaded.
For source reproducibility, also record `git rev-parse HEAD` in the checkout,
because different revisions can share a version number.

## Fit and evaluate a regression model

The response below depends on the first two of four predictors, plus
independent noise. Fit one direction using 120 observations and reserve the
remaining 40 for evaluation.

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
print("Reduced coordinates:", fit.transform(X[120:]).shape)
```

Observed output from a Windows source build:

```text
Converged: True Iterations: 10
Held-out MSE: 0.390764
Reduced coordinates: (40, 1)
```

The 40 predictions estimate mean responses; the observed responses also
contain noise. Their mean squared error is about 0.391, in squared response
units. The `(40, 1)` array contains one reduced coordinate per held-out row.
This single simulation is a use example, not a performance benchmark.
Last digits and iteration counts can differ with numerical libraries.

## Prepare your own data

Supply finite real arrays with observations in rows. `X` has shape `(n, p)`;
a response, event indicator, dose, or reward has shape `(n,)`. Encode
categorical variables and handle missing values before fitting. Estimate any
external preprocessing on training observations and reuse it for new data.
Regression uses one pooled predictor scale for optimization. SAVE retains
its supplied-unit estimating equation. Survival fitting standardizes predictors
internally.
All prediction methods accept new data in
the same coordinates supplied at fitting.

`keep_data` defaults to `False`, so request `True` when you need predictions.
`FitResult` supports both `fit.B` and `fit["B"]`. Dotted diagnostic names can
use underscores in attributes, for example `fit.stop_reason`.

Continue with [regression](regression.md), [survival](survival.md), or
[personalized doses](personalized-dose.md). The [API reference](../api.md)
contains the full defaults and return fields.
