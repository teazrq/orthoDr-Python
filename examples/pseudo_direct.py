"""Fit the reward surface first, then evaluate a continuous dose rule.

Run after installing orthoDr: python examples/pseudo_direct.py
The optimal dose is known here because the data are simulated.
"""
import numpy as np
from orthodr import orthoDr_pdose

rng = np.random.default_rng(1204)
X = rng.normal(size=(120, 4))
a = rng.uniform(0, 4, len(X))
optimal = 2 + .8 * np.tanh((X[:, 0] + X[:, 1]) / np.sqrt(2))
r = 3 * np.exp(-((a - optimal) / .7)**2) + rng.normal(0, .1, len(X))

# The conditional reward surface depends on dose and one predictor index.
# Opt into a separate unpenalized constant in the final ridge dose rule.
fit = orthoDr_pdose(
    X, a, r, method="pseudo_direct", ndr=1, lambda_init=[.05, .1, .3],
    intercept=True, K=12, keep_data=True, ncore=1,
)
print("Converged:", fit.converge, "Stop reason:", fit.stop_reason)
print("Normalized gradient norm:", fit.gradient_norm)
print("Reward fitting RMSE:", fit.fn, "Selected ridge penalty:", fit.lambda_init)

new_X = rng.normal(size=(1000, 4))
truth = 2 + .8 * np.tanh((new_X[:, 0] + new_X[:, 1]) / np.sqrt(2))
recommended = fit.predict(new_X)
dose_rmse = np.sqrt(np.mean((recommended - truth)**2))
value_regret = np.mean(3 - 3 * np.exp(-((recommended - truth) / .7)**2))
print("Dose RMSE against simulated truth:", dose_rmse)
print("Mean reward loss against the optimal simulated rule:", value_regret)

# Convergence describes numerical stationarity. The two held-out quantities
# assess decisions in this simulation; they do not establish accuracy for
# another population. Predictions are continuous and are not clipped.
