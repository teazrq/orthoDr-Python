"""Execute every documentation Python block, with a fresh process per page.

Blocks on the same page share a namespace, as they do when read in order.
Use --installed to check the shared installation without source-tree imports.
The fixed tutorial workflows also check the results their prose interprets.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
FENCE = re.compile(r"^```(?:python|py)\s*\n(.*?)^```\s*$", re.MULTILINE | re.DOTALL)


def python_blocks(path):
    source = path.read_text(encoding="utf-8")
    return [dict(source=match.group(1), line=source.count("\n", 0, match.start()) + 2)
            for match in FENCE.finditer(source)]


def validate_import(package_file, installed):
    actual = Path(package_file).resolve()
    if installed:
        expected = Path(importlib.metadata.distribution("orthodr").locate_file(
            "orthodr/__init__.py")).resolve()
        if actual.is_relative_to(ROOT) or actual != expected:
            raise ValueError(f"Expected shared installed package {expected}, imported {actual}")
    elif actual != (ROOT / "orthodr" / "__init__.py").resolve():
        raise ValueError(f"Expected source package {ROOT / 'orthodr'}, imported {actual}")
    return actual


def check_fit(fit, kind, method, shape):
    import numpy as np

    assert (fit.kind, fit.method) == (kind, method)
    assert fit.B.shape == shape and np.isfinite(fit.B).all()
    assert np.isfinite(fit.fn) and fit.bw > 0
    assert fit.keep_data, "Tutorial predictions require retained training data"
    assert fit.converge, f"{method} tutorial did not meet its stopping criterion"


def check_regression(namespace):
    import numpy as np

    fit, x, y, prediction = (namespace[key] for key in ("fit", "X", "y", "prediction"))
    check_fit(fit, "reg", "sir", (4, 1))
    assert prediction.shape == (40,) and np.isfinite(prediction).all()
    np.testing.assert_allclose(fit.transform(x[120:]), x[120:] @ fit.B)
    assert fit.transform(x[120:]).shape == (40, 1)
    assert np.isfinite(np.mean((prediction - y[120:]) ** 2))
    if "fit_from_start" in namespace:
        alternative = namespace["fit_from_start"]
        check_fit(alternative, "reg", "sir", (4, 1))
        prediction = namespace["prediction_from_start"]
        assert prediction.shape == (40,) and np.isfinite(prediction).all()


def check_survival(namespace):
    import numpy as np

    fit, curves = namespace["fit"], namespace["curves"]
    check_fit(fit, "surv", "dm", (4, 1))
    times = np.sort(namespace["time"][:110][namespace["event"][:110] == 1])
    np.testing.assert_array_equal(curves["timepoints"], times)
    survival = curves["surv"]
    assert survival.shape == (69, 30) and np.isfinite(survival).all()
    assert ((survival >= 0) & (survival <= 1)).all()
    assert (np.diff(survival, axis=0) <= 1e-12).all()


def check_direct(namespace):
    import numpy as np

    fit, recommended = namespace["fit"], namespace["recommended"]
    check_fit(fit, "pdose", "direct", (3, 1))
    assert fit.objective_version == "direct_joint_intercept_v2"
    assert fit.stop_reason == "gradient_tolerance"
    residuals = fit.gradient_norm
    assert {"B", "W", "intercept", "policy"} <= residuals.keys()
    assert all(np.isfinite(value) and 0 <= value <= 1e-6
               for value in residuals.values())
    np.testing.assert_allclose(fit.fn, -fit.value + fit.penalty)
    assert recommended.shape == (50,) and np.isfinite(recommended).all()
    assert np.isfinite(np.sqrt(np.mean((recommended - namespace["truth"]) ** 2)))
    np.testing.assert_allclose(fit.predict(namespace["X"]).mean(), fit.intercept,
                               rtol=1e-10, atol=1e-10)
    assert fit.intercept_bounds[0] <= fit.intercept <= fit.intercept_bounds[1]


def check_pseudo(namespace):
    import numpy as np

    fit, recommended = namespace["pseudo_fit"], namespace["pseudo_recommended"]
    check_fit(fit, "pdose", "pseudo_direct", (4, 1))
    assert fit.objective_version == "pseudo_mse_intercept_v1"
    assert fit.stop_reason == "gradient_tolerance" and 0 <= fit.gradient_norm <= 1e-6
    assert fit.lambda_init > 0 and fit.reward_scale > 0
    np.testing.assert_allclose(fit.fn, fit.reward_scale * np.sqrt(fit.objective_scaled))
    assert recommended.shape == (1000,) and np.isfinite(recommended).all()
    error = recommended - namespace["pseudo_truth"]
    rmse = np.sqrt(np.mean(error ** 2))
    loss = np.mean(3 - 3 * np.exp(-(error / .7) ** 2))
    assert np.isfinite(rmse) and 0 <= loss <= 3
    np.testing.assert_allclose(namespace["pseudo_dose_rmse"], rmse)
    np.testing.assert_allclose(namespace["pseudo_reward_loss"], loss)


def check_optimizer(namespace):
    import numpy as np

    result, covariance = namespace["result"], namespace["C"]
    basis = result["B"]
    assert result["converge"] and result["method"] == "true gradient"
    assert basis.shape == (3, 1) and np.isfinite(basis).all()
    np.testing.assert_allclose(basis.T @ basis, np.eye(1), rtol=0, atol=1e-10)
    objective = float(np.trace(basis.T @ covariance @ basis))
    np.testing.assert_allclose(result["fn"], objective, rtol=0, atol=1e-10)
    np.testing.assert_allclose(objective, np.linalg.eigvalsh(covariance)[-1],
                               rtol=0, atol=1e-8)
    assert np.isfinite(result["fn_Seq"][:result["itr"]]).all()


def check_page(page, namespace):
    stem = Path(page).stem
    if stem in {"index", "get-started", "regression"} and "fit" in namespace:
        check_regression(namespace)
        return "regression predictions and reduced coordinates"
    if stem == "survival":
        check_survival(namespace)
        return "survival event times, dimensions, probabilities and monotonicity"
    if stem == "personalized-dose":
        check_direct(namespace)
        check_pseudo(namespace)
        return "both dose objectives, convergence diagnostics and simulated decision errors"
    if stem == "general-optimizer":
        check_optimizer(namespace)
        return "orthogonality and independently computed maximum eigenvalue"
    return "all Python blocks executed"


def run_worker(payload):
    installed = payload["installed"]
    if installed:
        # Retain the shared user site, but remove this checkout and cwd.
        sys.path[:] = [path for path in sys.path
                       if path and not Path(path).resolve().is_relative_to(ROOT)]
    else:
        sys.path.insert(0, str(ROOT))
    import orthodr

    actual = validate_import(orthodr.__file__, installed)
    print(f"Page {payload['page']}: pid={os.getpid()}, package={actual}", flush=True)
    namespace = {"__name__": "__main__"}
    for block in payload["blocks"]:
        filename = f"{payload['page']}:{block['line']}"
        exec(compile(block["source"], filename, "exec"), namespace)
    print(f"Checked {len(payload['blocks'])} blocks: {check_page(payload['page'], namespace)}",
          flush=True)


def verify_docs(*, installed=False):
    pages = [(path, python_blocks(path)) for path in sorted((ROOT / "docs").rglob("*.md"))]
    pages = [(path, blocks) for path, blocks in pages if blocks]
    if not pages:
        raise ValueError("No executable Python documentation blocks were found")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    for path, blocks in pages:
        # Neither the process cwd nor PYTHONPATH may shadow the installed package.
        with tempfile.TemporaryDirectory(prefix="orthodr-docs-") as workdir:
            payload = dict(page=path.relative_to(ROOT).as_posix(), blocks=blocks,
                           installed=installed)
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker"],
                input=json.dumps(payload), text=True, capture_output=True,
                cwd=workdir, env=environment)
        print(result.stdout, end="", flush=True)
        if result.returncode:
            raise RuntimeError(f"{payload['page']} failed:\n{result.stderr}")
    mode = "installed" if installed else "source"
    count = sum(len(blocks) for _, blocks in pages)
    print(f"All {count} Python blocks passed across {len(pages)} documentation pages ({mode}).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed", action="store_true", help="Check the shared installed package")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.worker:
        run_worker(json.load(sys.stdin))
    else:
        try:
            verify_docs(installed=options.installed)
        except (ValueError, RuntimeError) as exc:
            parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
