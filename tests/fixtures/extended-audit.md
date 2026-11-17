# Extended numerical audit

`r_extended.json` is regenerated with seed `271828` from the clean canonical R
checkpoint recorded in `tools/r_source_manifest.json`. Its source fingerprints
must match `r_reference.json` and `r_pseudo.json`. The generator validates the
archive before loading any R code and after fitting, and rebuilds its native
objects with existing shared R libraries.

From the package root:

```text
Rscript tools/generate_r_extended.R PATH_TO_ISOLATED_PINNED_SOURCE tests/fixtures/r_extended.json
python -m pytest tests/test_r_extended.py -q
```

The optional third argument is a source manifest with the same exact pin.
It is independent of the previously shipped fixture, so regeneration never
requires changing old fixture provenance first.

The experiment fits 39 models covering every regression and survival method,
one and two directions where supported, automatic and explicit initializers,
direct and pseudo-direct dose models, and a two-direction direct intercept.
Thirteen representative cases are saved. Tests compare every saved fixed-state
prediction and all simple returned objectives. SEFF's nuisance reference remains
fixed at the initial projection, so its prediction is checked without replacing
its objective with a different recomputation.

Three complete two-direction direct fits have tight state and prediction
checks. Their fixed-state analytic derivatives and Hessians are also compared,
with central differences in every basis coordinate. Five exact R-preprocessed
native inputs separate fitting arithmetic from Python preprocessing. Inherited
regression and martingale routines are compared at the first supported step;
the analytic pseudo-direct routine is compared tightly at 1, 5, and 20 updates.
For the unconverged 100-update R state and a separate complete Python fit, tests
independently recompute each state's own objective, analytic constrained-gradient
norm, and ridge coefficients. They retain the strict fixed-R-state prediction
check without requiring loose agreement between the two long-run endpoints.

Pseudo-direct's automatic basis uses corrected pSAVE followed by ordered,
two-pass modified Gram-Schmidt in both languages. Its native prepared inputs
contain raw doses and grid values, without the obsolete exponentiated `a_dist`
argument. Direct learning retains that existing private argument.

The saved `sensitivity` section perturbs the first entry of the orthonormal
starting basis by `0`, `+/-1e-14`, `+/-1e-13`, `+/-1e-12`, and `+/-1e-11`, then
calls the public R wrapper. This records current behavior. Regression and
survival still use inherited finite differences; pseudo-direct now uses
analytic derivatives. Historical pseudo-direct finite-difference discrepancies
from the previous source revision are preserved in the implementation audit,
not described as properties of the revised solver.

The first Ubuntu CI run on Python 3.10 and 3.14 differed from the saved
two-direction SIR basis by `1.03e-7` after five updates. A separate check using
the pinned, compiled R source reproduced the saved state exactly, then
perturbed each starting-basis entry by `+/-1e-15`. R-only basis differences
reached `1.40e-7`, objective differences reached `1.08e-9`, and prediction
differences on the saved test observations reached `1.03e-6`. Thus even this
short inherited finite-difference trajectory can amplify rounding. The
five-update state remains unconverged, with constrained residual about
`0.00758` against `gtol=1e-6`. Independent SIR equations reproduce the returned
objectives within `1.7e-18`. Strict fixed-state objective and prediction checks
are retained; trajectory differences do not establish a different objective
or a stationary solution.

The Ubuntu objective difference is `7.96e-10`. At the same fixed R basis, its
objective differs by only `2.71e-19`; reevaluation at its returned basis agrees
exactly with its reported objective. The initial forward gradient differs by
`1.95e-12`, consistent with rounding amplified by subtraction and the subsequent
spectral updates. Orthonormality error is `2.22e-16`. The five-update SIR
comparison therefore uses absolute bounds of `2e-7` for the basis and `2e-9`
for the objective, covering the independently observed R-only sensitivity.
Other trajectory bounds are unchanged. Fixed-state objective and prediction
checks, returned-state objective consistency, orthonormality, iteration counts
and convergence flags remain checked.

On the audited Windows build, the 100-update public pseudo-direct predictions
differed by `4.0e-6`, with both fits stopping at `max_iterations`. Exact-R-start
native basis differences were zero at one update, `6.4e-14` at five, `3.9e-11`
at twenty, and `2.1e-6` at one hundred. R-only starting perturbations of
`+/-1e-14` changed predictions by up to `3.4e-5`; `+/-1e-11` reached `1.7e-4`.
Thus analytic derivatives greatly reduce the prior discrepancy but do not
eliminate all spectral-trajectory sensitivity. These nonstationary endpoints
are retained as diagnostic evidence rather than advertised as exact fit parity.

Only pseudo-direct and direct learning use their new explicit gradient-only
convergence contract. Regression and survival still permit their historical
small-change convergence stops, so their convergence flags alone do not certify
that the requested gradient tolerance was reached. Fixed-state objectives and
predictions remain tightly checked independently of that convention.

These experiments establish computational correspondence for the supplied
inputs. They do not establish global optimality, out-of-sample policy quality,
or identical long finite-difference paths on every numerical-library build.
