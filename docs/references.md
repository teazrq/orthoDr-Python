# References and citation

Cite the orthoDr software paper and the methodological papers relevant to
your analysis. The references below identify each paper's role; numerical
implementation details for this Python version are in the [API](api.md).

## Software

Zhu, R., Zhang, J., Zhao, R., Xu, P., Zhou, W., and Zhang, X. (2019).
[*orthoDr: Semiparametric Dimension Reduction via Orthogonality Constrained
Optimization*](https://journal.r-project.org/articles/RJ-2019-006/index.html).
*The R Journal*.

This paper describes the orthoDr software and its optimization framework.
Also record the [Python source revision](https://github.com/teazrq/orthoDr-Python)
used for an analysis, because the version number alone does not identify a
development checkout.

## Regression

Ma, Y., and Zhu, L. (2012).
[*A Semiparametric Approach to Dimension Reduction*](https://doi.org/10.1080/01621459.2011.646925).
*Journal of the American Statistical Association*, 107(497), 168-179.

Ma, Y., and Zhu, L. (2013).
[*Efficient Estimation in Sufficient Dimension Reduction*](https://doi.org/10.1214/12-AOS1072).
*The Annals of Statistics*.

These papers develop the semiparametric estimating-equation approach and
efficient estimation underlying the regression methods.

## Survival

Sun, Q., Zhu, R., Wang, T., and Zeng, D. (2019).
[*Counting process-based dimension reduction methods for censored outcomes*](https://doi.org/10.1093/biomet/asy064).
*Biometrika*, 106(1), 181-196.

This paper develops the counting-process dimension-reduction methods for
right-censored outcomes used by `orthoDr_surv`.

## Personalized dose learning

Zhou, W., Zhu, R., and Zeng, D. (2021).
[*A parsimonious personalized dose-finding model via dimension reduction*](https://doi.org/10.1093/biomet/asaa087).
*Biometrika*, 108(3), 643-659.

This paper develops direct and pseudo-direct personalized dose learning and
their dimension-reduction targets.

## Hazard MAVE

Xia, Y., Zhang, D., and Xu, J. (2010).
[*Dimension Reduction and Semiparametric Estimation of Survival Models*](https://doi.org/10.1198/jasa.2009.tm09372).
*Journal of the American Statistical Association*, 105(489), 278-290.

`hMave` uses the authors' hazard-based dimension reduction method for censored survival times.

## Partial SAVE

Feng, Z., Wen, X. M., Yu, Z., and Zhu, L. (2013).
[*On Partial Sufficient Dimension Reduction With Applications to Partially Linear Multi-Index Models*](https://doi.org/10.1080/01621459.2012.746065).
*Journal of the American Statistical Association*, 108(501), 237-246.

Feng and colleagues study partial sufficient dimension reduction with
continuous conditioning variables. In orthoDr, `pSAVE` supplies a starting
basis for personalized dose estimation.

## Orthogonality-constrained optimization

Wen, Z., and Yin, W. (2013).
[*A feasible method for optimization with orthogonality constraints*](https://doi.org/10.1007/s10107-012-0584-1).
*Mathematical Programming*.

This paper develops the feasible optimization approach used by the
orthogonality-constrained solvers. See the
[general optimizer guide](tutorials/general-optimizer.md) for Python callbacks.

## License and attribution

The Python distribution uses GPL version 3 or later. See the
[license](https://github.com/teazrq/orthoDr-Python/blob/main/LICENSE) and
[third-party notices](https://github.com/teazrq/orthoDr-Python/blob/main/THIRD_PARTY_NOTICES.md)
for the original authors and numerical dependencies.
