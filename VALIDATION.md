# Validation of the Python port

This development revision includes coordinated R/Python corrections to pseudo-direct learning and pSAVE. The numerical source is pinned to R source commit [`a9588d8`](https://github.com/teazrq/orthoDr/commit/a9588d8dd73c39188d1c46f371763addb10bee9a). The [fixture instructions](tests/fixtures/README.md) describe its source fingerprints and regeneration. R is needed for fixture generation, but not for Python fitting, prediction, or saved-fixture tests.

## Numerical agreement

The source-tree suite passes **271 tests**. Three freshly generated R fixture sets cover regression, survival, both dose methods, initializers, prediction, the general optimizer, and the revised pseudo-direct objective and intercept. Every fixture records the exact R source and inputs. All 44 pinned R source files and 19 mechanically adapted native files are checked against their hashes.

Source fingerprints normalize CRLF line endings to LF and preserve all other bytes. All 44 fingerprints match the canonical Git blobs, and all three fixture generators ran successfully from an exact archive of the pinned revision. Their numerical outputs are exactly unchanged from the preserved prior fixtures. Tests accept equivalent LF and CRLF checkouts while rejecting changed source content.

Independent formulas check the pseudo-direct analytic derivative, projected variance terms, actual Cayley slopes, stabilized dose-grid averages, intercept ridge coefficients and GCV. Across the ordinary fixed-state fixtures, maximum absolute differences are approximately `1.7e-16` for normalized objective, `2.3e-16` for gradient, `8.3e-15` for grid scores, and `4.4e-14` for coefficients. Extreme finite ridge penalties, exact constant targets, failed searches and both new prediction versions have separate regression tests.

Grid maxima and GCV minima are discrete choices. Tests compare their scores directly and distinguish well-separated winners from numerical near ties. For example, an identity kernel makes no-intercept GCV mathematically equal across penalties. Last-bit score differences can then select different penalties; coefficients are checked at the selected penalty rather than concealing the difference with a relaxed prediction tolerance.

Analytic gradients reduce the earlier pseudo-direct trajectory discrepancy, but nonconvex spectral updates still amplify rounding. In the preserved two-direction example, exact R-prepared inputs agree through 20 updates within `3.9e-11` in the basis. At 100 unfinished updates, public predictions differ by `4.0e-6`; the normalized constrained residual is about `0.00375` in both languages. R-only perturbations of the initial basis by `1e-14` change predictions by up to `3.3e-5`. These are unfinished trajectories, not convergence evidence. Tests retain tight early-step agreement, independently verify diagnostics at both final states, and compare predictions tightly with the fitted state held fixed. See the [extended numerical audit](tests/fixtures/extended-audit.md).

Inherited SIR finite differences are sensitive even over five updates: the first Ubuntu runs differed from the saved basis by `1.03e-7`, while R-only starting perturbations of `1e-15` produced basis differences of `1.40e-7` and objective differences of `1.08e-9`. That state is unfinished, with constrained residual about `0.00758`. Fixed-state objective and prediction checks remain strict. The [extended numerical audit](tests/fixtures/extended-audit.md) records the comparison.

## Convergence and compatibility

Pseudo-direct `converge=True` now requires the normalized constrained gradient tolerance. Five accepted updates with small average parameter and objective changes can instead report `stagnation`, with `converge=False`. The returned `fn` remains reward RMSE in its original units. Direct learning retains its joint policy/direction criterion. Regression and survival retain their previous small-change stopping rules.

Genuine earlier no-version pseudo-direct, direct, and direct-intercept objects preserve predictions exactly in both R and Python. New saved objects also pass fresh-process serialization checks. Public tests cover input preservation, array layouts, no-data retention, malformed intercept state, and empty, single-row and batched predictions.

pSAVE now pools group SAVE information in common coordinates and maps directions back to the supplied predictor units. Independent affine and covariance examples test that construction. Automatic starts for both dose methods can change. Coordinate correctness and numerical stationarity do not establish better decisions: small dose groups can have noisy covariance estimates, and the later finite-sample kernel depends on the chosen basis. Statistical comparisons must assess held-out dose error and reward loss separately.

## Runnable examples

`python tools/verify_examples.py` runs each of the three README workflows in a fresh process, checks its actual import path, and verifies convergence and prediction fields. It separately runs `examples/pseudo_direct.py`, checking its distinct normalized diagnostics and simulated decision metrics. The existing direct example checks remain intact.

| Example | Observed source-build output |
| --- | --- |
| SIR regression | Held-out MSE `0.390764`; 10 iterations |
| Martingale survival | 69 event-time rows by 30 held-out observations; 9 iterations |
| Direct dose, with intercept | Dose RMSE `0.081609`; gradient-tolerance stop |
| Pseudo-direct, with intercept | Dose RMSE `0.110015`; mean reward loss `0.071130`; gradient-tolerance stop |

These individual simulations illustrate use. They do not select a default penalty or establish real-data performance.

## Build coverage

Validation uses Windows, Python 3.14.7, NumPy 2.5.2, SciPy 1.18.1, shared Rtools45 and Armadillo 15.4.2 headers. The R reference uses R 4.6.1, dr 3.0.11 and pracma 2.4.6. Runtimes, dependencies and build tools come from existing shared installations. The extension does not link to an R runtime.

A fresh source archive built successfully and was installed into the shared
user Python installation. All **271 tests passed** from outside the source
tree against that installation, and all four examples passed in source and
installed modes. Import paths were checked explicitly. The matching R source
also passed `R CMD check` and its installed examples.

The [GitHub Actions workflow](https://github.com/teazrq/orthoDr-Python/actions/workflows/ci.yml) checks archive installation, the full installed suite, and all four installed examples on Ubuntu with Python 3.10 and 3.14. Its completed runs record the results for each commit. macOS builds have not been validated.

```bash
python -m pytest -q
python tools/verify_examples.py
python tools/verify_examples.py --installed
```

The Python source is published on [GitHub](https://github.com/teazrq/orthoDr-Python). No PyPI release or Python documentation site is published.
