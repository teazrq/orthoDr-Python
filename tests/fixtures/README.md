# R reference and numerical audit

The numerical fixtures are generated from the clean canonical R checkpoint
recorded in `tools/r_source_manifest.json`. This independent manifest pins every
file under `R/` and `src/`, plus `DESCRIPTION` and `NAMESPACE`. The native adapter
checks source SHA-256 fingerprints before regenerating `src/cpp`; fixture
generators check MD5 fingerprints before loading R and again before writing.
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
python -m pytest tests/test_r_parity.py tests/test_r_extended.py tests/test_r_pseudo.py tests/test_source_provenance.py -q
```

The generators use the existing shared R installation and user-wide packages
`pkgload`, `jsonlite`, `Rcpp`, `RcppArmadillo`, `dr`, and `pracma`. They rebuild
native objects with `pkgload::load_all` inside the isolated source snapshot;
they do not install over the user's orthoDr package. No project-local R library
or Python environment is used. The R session, package versions, seed, exact
source identifier, and source fingerprints are saved inside every fixture.
Matrices retain both dimensions.

The source manifest is intentionally separate from old fixtures. A source
revision change requires verifying a new exact archive and generating new
numerical results. Editing the source identifier in an old fixture does not
create a new oracle.

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

`r_pseudo.json` checks the revised pseudo-direct computation at fixed R states.
Python tests independently calculate the normalized objective, pairwise
derivatives including projected-variance derivatives, stable grid scores and
first-maximizer targets, intercept and ordinary ridge coefficients, explicit
smoother GCV, and both Cayley curves and their actual slopes. Cases include
multiple directions, unequal predictor scales, reward and dose affine changes,
scalar and vector ridges, huge ridges, narrow bandwidths, exact zero training
residuals, repeated `0.1` targets, skipped unusable ridge candidates, and accepted
state traces.

Both dose solvers now use analytic derivatives. Pseudo-direct's convergence
flag is true only for its normalized constrained-gradient tolerance. A small
parameter/objective change with a larger residual is a nonconverged stagnation
stop. Regression and survival retain their existing finite-difference routines
and stopping conventions. The short numerical experiments establish
implementation agreement on the saved inputs; they do not establish
statistical accuracy, global optima, or platform-independent trajectories.
