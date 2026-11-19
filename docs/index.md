# orthoDr for Python

Estimate a few linear combinations of continuous predictors for regression,
right-censored survival outcomes, or personalized dose learning. `orthodr`
fits the methods of the [orthoDr R package](https://teazrq.github.io/orthoDr/)
and predicts with NumPy arrays, without an R session.

[Get started: install and fit a model](tutorials/get-started.md){ .md-button .md-button--primary }

## Choose your analysis

| Task | Guide | Returned result |
| --- | --- | --- |
| Predict a continuous response | [Regression](tutorials/regression.md) | Directions and kernel estimates of the mean response |
| Model right-censored event times | [Survival](tutorials/survival.md) | Directions and conditional survival curves |
| Learn a continuous dose rule | [Personalized doses](tutorials/personalized-dose.md) | Directions, a dose rule, and numerical diagnostics |
| Optimize your own criterion | [General optimizer](tutorials/general-optimizer.md) | A matrix with orthonormal columns and its objective value |

Each model returns a direction matrix `fit.B` with one row per predictor and
`ndr` columns. `fit.transform(X)` gives the reduced coordinates `X @ fit.B`.
Set `keep_data=True` when fitting to retain the training information needed
for prediction. You choose `ndr`; these functions do not estimate the number
of directions automatically.

## Start with a reproducible example

The [first regression analysis](tutorials/get-started.md#fit-and-evaluate-a-regression-model)
holds out observations before fitting. The
[survival guide](tutorials/survival.md) explains event indicators and the
orientation of the returned survival matrix. The
[dose guide](tutorials/personalized-dose.md) distinguishes direct learning from
reward-surface fitting and shows what their convergence diagnostics mean.

These simulations illustrate use. A small training objective or a numerical
convergence flag does not establish accurate predictions or decisions on new
observations. The default bandwidth and dose penalties are starting choices,
not values calibrated for a new application.

## Source version and compatibility

These guides describe version **2.1.0**. The source build requires a C++17
compiler, Armadillo, BLAS, and LAPACK. See
[installation](tutorials/get-started.md#install-from-source) for shared Python
dependencies and Windows/Linux build prerequisites.

The package uses the R package's C++ numerical routines and implements
preprocessing, initializers, and prediction in Python.
The [Python and R comparison](reference/r-python.md) explains what was checked,
where numerical trajectories can differ, and how the interfaces correspond.
It does not claim uniform numerical agreement across all data and platforms.

The [API reference](api.md) gives signatures, defaults, return fields, and
utilities. See [references and citation](references.md) for the method papers
and software attribution. The distribution uses GPL version 3 or later.
