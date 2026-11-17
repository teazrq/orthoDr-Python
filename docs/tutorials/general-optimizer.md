# General optimizer

Optimize a scalar Python objective over a matrix with orthonormal columns,
using `ortho_optim`. Supply an analytic gradient or let the solver approximate
one by forward differences.

## Express the objective and gradient

The feasible matrices satisfy $B^\top B=I$. As a small example, find the
one-dimensional direction explaining the most variance for a diagonal
covariance matrix $C$:

$$
\max_{B^\top B=1}\;\operatorname{tr}(B^\top C B),
\qquad C=\operatorname{diag}(5,2,1).
$$

The Euclidean gradient is $2CB$. The largest eigenvalue of $C$ is 5, so this
example has a known reference value. Its solution is a direction along the
first coordinate, with either sign.

```python
import numpy as np
from orthodr import ortho_optim

C = np.diag([5.0, 2.0, 1.0])
B_initial = np.ones((3, 1)) / np.sqrt(3)

def variance(B, covariance):
    return float(np.trace(B.T @ covariance @ B))

def variance_gradient(B, covariance):
    return 2 * covariance @ B

result = ortho_optim(
    B_initial, variance, variance_gradient, C, maximize=True,
)
print("Converged:", result["converge"])
print(f"Explained variance: {result['fn']:.6f}")
print(f"Constraint error: {np.linalg.norm(result['B'].T @ result['B'] - np.eye(1)):.2e}")
```

Observed output from a Windows source build:

```text
Converged: True
Explained variance: 5.000000
Constraint error: 0.00e+00
```

The objective reaches the known maximum of 5. The constraint error is the
Frobenius norm of `B.T @ B - I`; its displayed zero means no error was visible
at the reported precision in this run. Other numerical environments can
return a small nonzero roundoff error.

## Adapt the callbacks to your criterion

`ortho_optim` minimizes by default. Set `maximize=True` for maximization;
returned objective values retain your objective's original sign. Additional
positional and keyword arguments are passed to both callbacks, as `C` is in
the example.

`fn` must return one finite scalar at every evaluation. An analytic `grad`
must return a finite matrix with the same shape as `B`. Supply the ordinary
Euclidean gradient of your objective; the solver constructs the constrained
update. If `grad=None`, forward differences use `control["epsilon"]`, whose
default is `1e-6`.

The starting matrix must have at least as many rows as columns and full column
rank. Its columns are orthonormalized when needed. This solver implements the
[Wen and Yin optimization approach](../references.md#orthogonality-constrained-optimization);
it does not guarantee a global optimum for a general nonconvex criterion.

## Inspect the returned state

The returned dictionary contains the fitted `B`, objective `fn`, iteration
count `itr`, stopping flag `converge`, and gradient `method`. Unlike the model
fits, it is an ordinary dictionary, so use `result["B"]` rather than
`result.B`.

`fn_Seq` stores completed-iteration objectives followed by allocated zeros.
Use `result["fn_Seq"][:result["itr"]]` for the trajectory. Convergence can
mean a small constrained gradient or small parameter and objective changes;
inspect your criterion and constraints as well as the flag. Reaching the
iteration limit does not establish convergence.

See the [API reference](../api.md#initializers-and-general-optimization) for
controls and the other numerical utilities.
