"""Check README workflows in fresh processes, from source or with --installed."""
import argparse
import ast
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ("reg", "surv", "pdose")


def read_examples(readme):
    examples = re.findall(r"```python\n(.*?)```", readme, flags=re.DOTALL)
    if len(examples) != len(WORKFLOWS):
        raise ValueError("README must contain the three regression, survival, and dose examples")
    for kind, source in zip(WORKFLOWS, examples):
        calls = {node.func.id for node in ast.walk(ast.parse(source))
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        if "orthoDr_" + kind not in calls:
            raise ValueError(f"README is missing the {kind} workflow in its expected example")
    return examples


def validate_import(package_file, installed):
    actual = Path(package_file).resolve()
    if installed:
        if actual.is_relative_to(ROOT):
            raise ValueError(f"Installed check imported source-tree package {actual}")
        expected = Path(importlib.metadata.distribution("orthodr").locate_file(
            "orthodr/__init__.py")).resolve()
        if actual != expected:
            raise ValueError(f"Installed check imported {actual}, expected installed package {expected}")
    elif actual != (ROOT / "orthodr" / "__init__.py").resolve():
        raise ValueError(f"Source check imported {actual}, expected {ROOT / 'orthodr'}")
    return actual


def check_example(payload):
    installed, kind = payload["installed"], payload["kind"]
    if installed:
        # Keep the shared user site, but exclude source-tree imports and cwd.
        sys.path[:] = [p for p in sys.path
                       if p and not Path(p).resolve().is_relative_to(ROOT)]
    else:
        sys.path.insert(0, str(ROOT))
    import os
    import orthodr

    actual = validate_import(orthodr.__file__, installed)
    mode = "installed" if installed else "source"
    print(f"Example {kind}: mode={mode}, pid={os.getpid()}, package={actual}", flush=True)
    namespace = {"__name__": "__main__"}
    exec(compile(payload["source"], f"README {kind} example", "exec"), namespace)
    fit, np = namespace["fit"], namespace["np"]
    expected_kind = "pdose" if kind == "pseudo" else kind
    assert fit.kind == expected_kind, f"Expected a {expected_kind} fit"
    assert np.isfinite(fit.B).all() and np.isfinite(fit.fn)
    if kind != "pseudo":
        assert fit.converge, f"{kind} README example did not converge"
    if kind == "reg":
        prediction = namespace["prediction"]
        assert prediction.shape == (40,) and np.isfinite(prediction).all()
    elif kind == "surv":
        curves = namespace["curves"]
        assert curves["surv"].shape == (len(curves["timepoints"]), 30)
        assert np.isfinite(curves["surv"]).all()
        assert ((curves["surv"] >= 0) & (curves["surv"] <= 1)).all()
        assert (np.diff(curves["surv"], axis=0) <= 1e-12).all()
    elif kind == "pseudo":
        recommended = namespace["recommended"]
        assert recommended.shape == (1000,) and np.isfinite(recommended).all()
        assert fit.method == "pseudo_direct"
        assert fit.objective_version == "pseudo_mse_intercept_v1"
        assert np.isclose(fit.fn, fit.reward_scale * np.sqrt(fit.objective_scaled))
        assert fit.converge == (fit.gradient_norm <= 1e-6)
        assert (fit.stop_reason == "gradient_tolerance") == fit.converge
        assert np.isfinite(namespace["dose_rmse"]) and np.isfinite(namespace["value_regret"])
    else:
        recommended = namespace["recommended"]
        assert recommended.shape == (50,) and np.isfinite(recommended).all()
        assert np.isclose(fit.fn, -fit.value + fit.penalty)
        assert fit.stop_reason == "gradient_tolerance"
        residuals = fit.gradient_norm
        required = {"B", "W"}
        if fit.get("objective.version") == "direct_joint_intercept_v2":
            required |= {"intercept", "policy"}
        assert required <= residuals.keys(), "Missing direct convergence residuals"
        # The README uses the documented default gtol, without a control override.
        assert all(np.isfinite(v) and 0 <= v <= 1e-6 for v in residuals.values()), (
            "Direct README residuals exceed its default gtol=1e-6")


def verify_readme(readme=None, *, installed=False):
    if readme is None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
    examples = read_examples(readme)
    outputs = []
    # The working directory cannot shadow the shared installed package.
    with tempfile.TemporaryDirectory(prefix="orthodr-examples-") as workdir:
        for kind, source in zip(WORKFLOWS, examples):
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker"],
                input=json.dumps(dict(kind=kind, source=source, installed=installed)),
                text=True, capture_output=True, cwd=workdir)
            print(result.stdout, end="", flush=True)
            if result.returncode:
                raise RuntimeError(f"{kind} example failed:\n{result.stderr}")
            outputs.append(result.stdout)
    mode = "installed" if installed else "source"
    print(f"All three README workflows converged and passed ({mode}).")
    return outputs


def verify_pseudo_example(*, installed=False):
    source = (ROOT / "examples" / "pseudo_direct.py").read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="orthodr-pseudo-example-") as workdir:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker"],
            input=json.dumps(dict(kind="pseudo", source=source, installed=installed)),
            text=True, capture_output=True, cwd=workdir)
    print(result.stdout, end="", flush=True)
    if result.returncode:
        raise RuntimeError(f"Pseudo-direct example failed:\n{result.stderr}")
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed", action="store_true", help="Check the shared installed package")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.worker:
        check_example(json.load(sys.stdin))
    else:
        try:
            verify_readme(installed=options.installed)
            verify_pseudo_example(installed=options.installed)
        except (ValueError, RuntimeError) as exc:
            parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
