# R reference and numerical audit

The numerical fixtures use the canonical R checkpoint recorded in
`tools/r_source_manifest.json`. This independent manifest pins every
file under `R/` and `src/`, plus `DESCRIPTION` and `NAMESPACE`. The native adapter
checks source SHA-256 fingerprints before regenerating `src/cpp`. Generators
for the pinned R fixtures check MD5 fingerprints before loading R and again before writing.
All pinned text fingerprints normalize CRLF line endings to LF. An exact Git
archive and a Windows checkout therefore identify the same source; any other
source change still fails verification.

Download and extract the
[source archive for commit `a9588d8`](https://github.com/teazrq/orthoDr/archive/a9588d8dd73c39188d1c46f371763addb10bee9a.tar.gz)
from the canonical R repository. Use the extracted package directory as
`PATH_TO_PINNED_ARCHIVE`. From the Python package directory, run:

```text
Rscript tools/generate_r_fixtures.R PATH_TO_PINNED_ARCHIVE tests/fixtures/r_reference.json
Rscript tools/generate_r_extended.R PATH_TO_PINNED_ARCHIVE tests/fixtures/r_extended.json
Rscript tools/generate_r_pseudo.R PATH_TO_PINNED_ARCHIVE tests/fixtures/r_pseudo.json
Rscript tools/generate_r_seff_centered.R PATH_TO_PINNED_ARCHIVE tests/fixtures/r_seff_centered.json
Rscript tools/generate_r_sir_scaled.R PATH_TO_PINNED_ARCHIVE tests/fixtures/r_sir_scaled.json
python -m pytest tests/test_r_parity.py tests/test_r_extended.py tests/test_r_pseudo.py tests/test_source_provenance.py -q
```

The generators use the existing shared R installation and user-wide packages
`pkgload`, `jsonlite`, `Rcpp`, `RcppArmadillo`, `dr`, and `pracma`. They rebuild
native objects with `pkgload::load_all` inside the isolated source snapshot;
they do not install over the user's orthoDr package. No project-local R library
or Python environment is used. The R session, package versions, seed, exact
source identifier, and source fingerprints are saved inside every fixture.
Matrices retain both dimensions.

The current native build includes the revised regression, survival, and dose calculations. To regenerate it,
run `python tools/vendor_core.py PATH_TO_REVISED_R_SRC --native-overlay`
with the [R 2.1.0 source directory](https://github.com/teazrq/orthoDr/tree/001bb696b280254ec26d197a46c4aeeef842c226/src). The adapter verifies the revised
files against `tools/r_native_source_overlay.json` and other native files against the
baseline. Historical R fixtures retain their original source fingerprints.

`r_reference.json` covers all regression and survival methods, ties, one event,
one predictor, direct and pseudo-direct dose models with and without an
intercept, initializers including corrected pSAVE, optimizer branches,
fixed-state prediction, and serialization. It retains the sensitive seven-step
martingale survival example and its R-only perturbation study. Its inherited
finite-difference trajectory checks keep their existing tolerances, including
strict first-step and fixed-state prediction checks.

`r_extended.json` adds an independent seed, two-direction fits, exact native
preprocessing, longer analytic pseudo-direct trajectories, complete direct fits,
and perturbation studies of inherited regression/survival routines. Details and
limits are described in `extended-audit.md`.

`r_seff_centered.json` records the earlier public SEFF expectations for the
explicit one- and two-direction fits and automatic two-direction fit already
in `r_reference.json`. Its generator calls the verified pinned R wrapper on
column-centered training predictors, then restores the original predictors
before prediction. This is the corrected public preprocessing expressed using
an immutable R oracle. Raw-input objective and initializer comparisons, and prediction
from the old fixed fitted states, still use their original expectations.

`r_sir_scaled.json` records the earlier public SIR results for the supplied one- and
two-direction starts and the stationary fit in `r_reference.json`. Its generator
calls the verified pinned R wrapper on centered predictors divided by their pooled
sample SD. It converts `fn` to the supplied predictor units and restores the
original predictors before prediction. Tests also compare the revised native
SAVE covariance calculations to independent fixed-basis moment equations.

`r_regression_scaled.json` records the corresponding built R development
tree's public regression fits on the same inputs, including automatic LOCAL
and SEFF fits and the stationary SIR case. It checks the common-scale optimization,
PHD centering, and SAVE's numerical objective scaling. The reference records
the R regression source fingerprints and reuses the prediction inputs from
`r_reference.json`. The pinned fixtures continue to support the native objective
and fixed-state prediction comparisons.

```text
Rscript tools/generate_r_regression_scaled.R PATH_TO_BUILT_R_TREE tests/fixtures/r_regression_scaled.json
```

`r_cp_sir_left.json` supplies the current CP-SIR directions for the tied-time,
one-event, and one-predictor inputs in `r_reference.json`. It also stores the
one-step and seven-step states and predictions for that fixture's automatic
two-direction martingale fit. Tests check singular values against both R
generations and prediction arithmetic at the fixed seven-step state. A saved
R-only perturbation study supports the wider projector and prediction bounds
for the seven-step comparison. The extended survival cases still check
objective and prediction arithmetic at historical fitted states.

To regenerate this historical CP-SIR fixture, copy the pinned archive to a
separate directory and apply the supplied patch to that copy. The generator
verifies the patched initializer's fingerprint and every other baseline file.

```text
git -C PATH_TO_CP_SIR_SNAPSHOT apply ABSOLUTE_PATH_TO_PYTHON_REPO/tools/cp_sir_left.patch
Rscript tools/generate_r_cp_sir_left.R PATH_TO_CP_SIR_SNAPSHOT tests/fixtures/r_cp_sir_left.json
```

`r_pseudo.json` records the historical pseudo-direct computation at fixed R states.
Python tests independently calculate the normalized objective, pairwise
derivatives including projected-variance derivatives, stable grid scores and
targets using the first grid position whose upper numerical endpoint reaches the largest
lower endpoint, intercept and ordinary ridge coefficients, explicit
smoother GCV, and both Cayley curves and their actual slopes. Cases include
multiple directions, unequal predictor scales, reward and dose affine changes,
scalar and vector ridges, huge ridges, narrow bandwidths, exact zero training
residuals, repeated `0.1` targets, skipped unusable ridge candidates, and accepted
state traces.

`r_pseudo_grid_ties.json` records the grid targets, ridge results, and training
predictions for the `bw_0.05` case using the revised R tie rule. In that case,
training row 4 changes from dose `1.025` to `0.5625` and the selected penalty
stays at `0.03`. The other 17 cases use the pinned expectations. The generator
uses `digest` from the shared R library and verifies the built R tree's native
source fingerprints against `src/cpp/provenance.json`.

```text
Rscript tools/generate_r_pseudo_grid_ties.R PATH_TO_BUILT_R_TREE tests/fixtures/r_pseudo_grid_ties.json
```

Both dose solvers now use analytic derivatives. Pseudo-direct's convergence
flag is true only for its normalized constrained-gradient tolerance. A small
parameter/objective change with a larger residual is a nonconverged stagnation
stop. Regression and survival retain their existing finite-difference routines
and stopping conventions. The short numerical experiments establish
implementation agreement on the saved inputs; they do not establish
statistical accuracy, global optima, or platform-independent trajectories.
