# Validation of the Python port

The native numerical source uses R source commit [`a9588d8`](https://github.com/teazrq/orthoDr/commit/a9588d8dd73c39188d1c46f371763addb10bee9a) as its baseline and incorporates fingerprinted updates to regression, survival optimization, and dose fitting and prediction. Python fitting, prediction, and saved-fixture tests run without R. R fixture regeneration uses the source fingerprints in the [fixture instructions](tests/fixtures/README.md).

## Numerical agreement

The R reference sets cover regression, survival, both dose methods, initializers, prediction, and the general optimizer. The current regression reference uses the revised R fitting functions on the original inputs. It includes the reported objective in the supplied units and predictions from each fit. Historical SIR and SEFF references preserve earlier comparisons. Each reference records its source and its inputs or the fixture that supplies them. The provenance checks verify 44 pinned R source files and 20 current native files.

CP-SIR now uses the left singular vectors of the existing estimation matrix. Python agrees with the revised R source on the identifiable leading subspaces for tied observations and one event, the single-predictor shape, and the singular values before and after the switch. For the two-direction automatic martingale fit, the fitted subspace and held-out survival predictions agree within `1e-15` after one update on Windows. After seven updates they differ by `1.2e-5` and `2.3e-5`, respectively. In R alone, changing one starting coefficient by `1e-14` moves these quantities by up to `2.0e-4` and `3.7e-4`. The tests use bounds informed by that finite-difference sensitivity for the seven-update fit and check predictions from a fixed R state within `2e-12` absolute and relative tolerance. The supplemental fixture records the revised `R/CP_SIR.R` fingerprint over the pinned source archive.

Source fingerprints normalize CRLF line endings to LF and preserve all other bytes. All 44 baseline fingerprints match the canonical Git blobs. The original reference, extended, and pseudo-direct generators reproduced their preserved numerical outputs from an exact archive of the pinned revision. Tests accept equivalent LF and CRLF checkouts while rejecting changed source content.

Independent formulas check the pseudo-direct analytic derivative, projected variance terms, actual Cayley slopes, stabilized dose-grid averages, intercept ridge coefficients and GCV. Across the ordinary fixed-state fixtures, maximum absolute differences are approximately `1.7e-16` for normalized objective, `2.3e-16` for gradient, `8.3e-15` for grid scores, and `4.4e-14` for coefficients. Extreme finite ridge penalties, exact constant targets, failed searches and both new prediction versions have separate regression tests.

For each pseudo-direct training target, both packages compare estimated grid rewards using compensated sums. For dose $j$, the comparison radius is $r_j=4\epsilon M_j$, where $\epsilon=2^{-52}$ and $M_j$ is the local weighted mean of the absolute standardized rewards. With estimated reward $s_j$, the fit selects the first grid point for which $s_j+r_j\geq\max_k(s_k-r_k)$. The public grid is in increasing dose order.

A row-permutation test checks the selected targets and the resulting continuous predictions. In the archived `bw_0.05` example, one target changed from 1.025 to 0.5625 and its fitted dose changed from 0.99515 to 0.54612. The two standardized reward scores agreed within machine precision. [Revised R reference values](tests/fixtures/r_pseudo_grid_ties.json)

GCV is a separate discrete choice. For example, an identity kernel makes no-intercept GCV mathematically equal across penalties. Last-bit differences can select different penalties across numerical libraries. The checks compare GCV values and verify the coefficients at the selected penalty.

Analytic gradients reduce the earlier pseudo-direct trajectory discrepancy, but nonconvex spectral updates still amplify rounding. In the preserved two-direction example, exact R-prepared inputs agree through 20 updates within `3.9e-11` in the basis. At 100 unfinished updates, public predictions differ by `4.0e-6`; the normalized constrained residual is about `0.00375` in both languages. R-only perturbations of the initial basis by `1e-14` change predictions by up to `3.3e-5`. These are unfinished trajectories, not convergence evidence. Tests retain tight early-step agreement, independently verify diagnostics at both final states, and compare predictions tightly with the fitted state held fixed. See the [extended numerical audit](tests/fixtures/extended-audit.md).

Inherited SIR finite differences are sensitive even over five updates: the first Ubuntu runs differed from the saved basis by `1.03e-7`, while R-only starting perturbations of `1e-15` produced basis differences of `1.40e-7` and objective differences of `1.08e-9`. That state is unfinished, with constrained residual about `0.00758`. Fixed-state objective and prediction checks remain strict. The [extended numerical audit](tests/fixtures/extended-audit.md) records the comparison.

In the unfinished seven-update two-direction LOCAL fit, Ubuntu Python 3.10 differed from the R fixture by `1.26e-5` in the basis. Using the basis printed in CI gives differences of `6.78e-6` in the subspace projector, `5.97e-6` in held-out predictions, and `3.40e-10` in the objective. Across all R-only sign perturbations of size `1e-15` in the starting coefficients, the largest pairwise differences were `1.96e-5`, `1.06e-5`, `9.31e-6`, and `5.38e-10`, respectively. The projected gradient norm was about `4.01e-4` at the saved R and CI states. The test bounds this trajectory variation for this case and independently checks the LOCAL objective and Gaussian predictions at each fitted basis.

## Convergence and compatibility

Pseudo-direct `converge=True` now requires the normalized constrained gradient tolerance. Five accepted updates with small average parameter and objective changes can instead report `stagnation`, with `converge=False`. The returned `fn` remains reward RMSE in its original units. Direct learning retains its joint policy/direction criterion. Regression and survival retain their previous small-change stopping rules.

Genuine earlier no-version pseudo-direct, direct, and direct-intercept objects preserve predictions exactly in both R and Python. New saved objects also pass fresh-process serialization checks. Public tests cover input preservation, array layouts, no-data retention, malformed intercept state, and empty, single-row and batched predictions.

pSAVE now pools group SAVE information in common coordinates and maps directions back to the supplied predictor units. Independent affine and covariance examples test that construction. Automatic starts for both dose methods can change. Coordinate correctness and numerical stationarity do not establish better decisions: small dose groups can have noisy covariance estimates, and the later finite-sample kernel depends on the chosen basis. Statistical comparisons must assess held-out dose error and reward loss separately.

SEFF subtracts training column means before automatic initialization and fitting, including with a custom start. Four regression tests check one- and two-direction fits under exactly representable predictor translations, with both default and custom starts. Directions, objectives, iteration counts, and convergence flags match exactly within each pair; held-out predictions agree within `2e-13`. The tests also check raw retained predictors and pickle predictions. Exactly representable inputs isolate this translation property from the finite-difference solver's sensitivity to last-bit input changes.

Scalar regression defaults to `[[1]]` before sliced initialization, while supplied starts retain their orientation. All five methods are checked with 2, 12, and 40 observations, omitted and capped `ndr`, and positive and negative starts. Each fit calls its usual solver once and converges with zero iterations. Independent Gaussian weighted-mean predictions agree within `1e-12`; a separate scalar SIR formula checks its reported objective. Invalid single-row, constant, and nonfinite data and a zero supplied basis are rejected. The existing multivariable tests remain unchanged.

Survival surfaces use observed follow-up order statistics at the 0th through 100th empirical percentiles, plus every observed event time, with duplicate knots removed. Exact integer ranks agree with the R type 1 convention at percentile boundaries. Tests inspect the coordinates and heights passed to Matplotlib, including unsorted samples of 1,000 and 1,001 observations, tied event/censoring times, times below one, time-unit changes, and every named transformation. Independent weighted Kaplan-Meier formulas agree within `1e-14`. A vector-dependent callable is evaluated once on the full input; input times remain unchanged. When all plotted times coincide, a rendered 3D line shows the actual survival profile at that single time, including all-event and all-censored cases.

## Runnable examples

`python tools/verify_examples.py` runs each of the three README workflows in a fresh process, checks its actual import path, and verifies convergence and prediction fields. It separately runs `examples/pseudo_direct.py`, checking its distinct normalized diagnostics and simulated decision metrics. The existing direct example checks remain intact.

| Example | Observed source-build output |
| --- | --- |
| SIR regression | Held-out MSE `0.390764`; 10 iterations |
| Martingale survival | 69 event-time rows by 30 held-out observations; 11 iterations |
| Direct dose, with intercept | Dose RMSE `0.081609`; gradient-tolerance stop |
| Pseudo-direct, with intercept | Dose RMSE `0.110015`; mean reward loss `0.071130`; gradient-tolerance stop |

These individual simulations illustrate use. They do not select a default penalty or establish real-data performance.

## Build coverage

Validation uses Windows, Python 3.14.7, NumPy 2.5.2, SciPy 1.18.1, shared Rtools45 and Armadillo 15.4.2 headers. The R reference uses R 4.6.1, dr 3.0.11 and pracma 2.4.6. Runtimes, dependencies and build tools come from existing shared installations. The extension does not link to an R runtime.

Before the SEFF centering correction, a fresh source archive built successfully
and was installed into the shared user Python installation. All **271 tests passed** from outside the source
tree against that installation, and all four examples passed in source and
installed modes. Import paths were checked explicitly. The matching R source
also passed `R CMD check` and its installed examples.

The [GitHub Actions workflow](https://github.com/teazrq/orthoDr-Python/actions/workflows/ci.yml) checks archive installation, the full installed suite, and all four installed examples on Ubuntu with Python 3.10 and 3.14. Its completed runs record the results for each commit. macOS builds have not been validated.

```bash
python -m pytest -q
python tools/verify_examples.py
python tools/verify_examples.py --installed
```

The Python source is published on [GitHub](https://github.com/teazrq/orthoDr-Python).
The [documentation site](https://teazrq.github.io/orthoDr-Python/) contains
installation instructions, tutorials, and the API guide. Its Python examples
are checked in fresh processes against the installed package with
`python tools/verify_docs.py --installed`; all nine current code blocks also
pass against the source checkout with `python tools/verify_docs.py`, including
the survival-surface example. The documentation build checks
internal links and section anchors. No PyPI release has been published.
