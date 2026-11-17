"""The README check must detect omitted workflows and unsuccessful fitting."""
import importlib.util
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verify_examples", ROOT / "tools" / "verify_examples.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)
README = (ROOT / "README.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("readme", ["", README.replace("```python", "```text", 1),
    README.replace("orthoDr_surv(", "orthoDr_reg(")])
def test_missing_or_replaced_workflow_is_not_success(readme):
    with pytest.raises(ValueError, match="three|missing"):
        verifier.read_examples(readme)


def test_initialization_only_direct_example_fails_convergence():
    changed = README.replace("keep_data=True, maxitr=100,", "keep_data=True, maxitr=0,")
    assert changed != README
    with pytest.raises(RuntimeError, match="pdose README example did not converge"):
        verifier.verify_readme(changed)


def test_convergence_label_does_not_hide_large_direct_residual():
    direct = verifier.read_examples(README)[2]
    changed = README.replace(direct, direct + '\nfit["gradient.norm"]["B"] = 1e-3\n')
    with pytest.raises(RuntimeError, match="residuals exceed.*gtol"):
        verifier.verify_readme(changed)


@pytest.mark.parametrize("installed", [False, True])
def test_workflows_use_fresh_processes_and_the_requested_package(installed):
    if installed:
        # Source-only contributors can run the rest without installing a wheel.
        result = subprocess.run([sys.executable, "-c", "import orthodr; print(orthodr.__file__)"],
            cwd=Path(sys.executable).parent, capture_output=True, text=True)
        if result.returncode or Path(result.stdout.strip()).resolve().is_relative_to(ROOT):
            pytest.skip("A shared orthodr installation is needed for its integration check")
    outputs = verifier.verify_readme(installed=installed)
    pids = [re.search(r"pid=(\d+)", output).group(1) for output in outputs]
    assert len(set(pids)) == 3
    for output in outputs:
        location = Path(re.search(r"package=(.*)", output).group(1)).resolve()
        assert location.is_relative_to(ROOT) is not installed
        assert f"mode={'installed' if installed else 'source'}" in output


def test_import_location_validation_rejects_source_as_installed():
    with pytest.raises(ValueError, match="Installed check"):
        verifier.validate_import(ROOT / "orthodr" / "__init__.py", installed=True)


def test_import_location_validation_rejects_other_package_as_source(tmp_path):
    with pytest.raises(ValueError, match="Source check"):
        verifier.validate_import(tmp_path / "orthodr" / "__init__.py", installed=False)
