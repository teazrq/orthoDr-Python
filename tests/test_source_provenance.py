"""Source regeneration must not label modified numerical code as the R baseline."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("vendor_core", ROOT / "tools/vendor_core.py")
VENDOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VENDOR)


def test_all_vendored_numerical_files_match_recorded_generated_hashes():
    expected = json.loads((ROOT / "src/cpp/provenance.json").read_text())
    actual = {p.name for p in (ROOT / "src/cpp").iterdir() if p.suffix in {".h", ".cpp"}}
    assert actual == set(expected["files"])
    for name, hashes in expected["files"].items():
        assert hashlib.sha256((ROOT / "src/cpp" / name).read_bytes()).hexdigest() == hashes["generated_sha256"], name


def test_source_pin_generated_manifest_fixtures_and_extension_agree():
    from orthodr import _core

    pin = json.loads(VENDOR.SOURCE_MANIFEST.read_text())
    overlay = json.loads(VENDOR.NATIVE_OVERLAY.read_text())
    generated = json.loads((ROOT / "src/cpp/provenance.json").read_text())
    assert pin["source_sha"] == VENDOR.SOURCE_REVISION == generated["base_revision"]
    assert _core.source_base_revision == overlay["base_revision"] == generated["base_revision"]
    assert set(generated["source_overlay"]) == set(overlay["files"])
    fingerprint = hashlib.sha256(json.dumps(overlay["files"], sort_keys=True).encode()).hexdigest()
    assert generated["revision"] == _core.source_revision == pin["source_sha"] + "+native." + fingerprint
    assert pin["source_hash_convention"] == generated["source_hash_convention"] == (
        "CRLF normalized to LF; all other bytes unchanged")
    assert set(pin["source_files_md5"]) == set(pin["source_files_sha256"])
    for name, hashes in generated["files"].items():
        baseline = pin["source_files_sha256"].get("src/" + name)
        if name in overlay["files"]:
            assert hashes == overlay["files"][name]
            assert hashes["source_sha256"] != baseline
        else:
            assert hashes["source_sha256"] == baseline
    for name in ("r_reference.json", "r_extended.json", "r_pseudo.json",
                 "r_seff_centered.json", "r_sir_scaled.json"):
        provenance = json.loads((ROOT / "tests/fixtures" / name).read_text())["provenance"]
        assert provenance["source_sha"] == pin["source_sha"], name
        assert provenance["source_files_md5"] == pin["source_files_md5"], name
        assert provenance["source_hash_convention"] == pin["source_hash_convention"], name


def test_pseudo_grid_fixture_identifies_its_r_solver():
    generated = json.loads((ROOT / "src/cpp/provenance.json").read_text())
    provenance = json.loads((ROOT / "tests/fixtures/r_pseudo_grid_ties.json").read_text())["provenance"]
    assert provenance["source_fixture"] == "r_pseudo.json"
    assert provenance["base_source_sha"] == generated["base_revision"]
    assert provenance["source_hash_convention"] == generated["source_hash_convention"]
    assert provenance["pseudo_solver_sha256"] == generated["files"]["pdose_semi_solver.cpp"]["source_sha256"]


def test_cp_sir_fixture_fingerprints_isolate_the_r_initializer_change():
    pin = json.loads(VENDOR.SOURCE_MANIFEST.read_text())
    provenance = json.loads((ROOT / "tests/fixtures/r_cp_sir_left.json").read_text())["provenance"]
    assert provenance["base_source_sha"] == pin["source_sha"]
    assert provenance["source_hash_convention"] == pin["source_hash_convention"]
    assert provenance["source_fixture"] == "r_reference.json"
    expected = pin["source_files_md5"]
    actual = provenance["source_files_md5"]
    assert set(actual) == set(expected)
    assert provenance["source_overlay"] == "R/CP_SIR.R"
    assert {name for name in actual if actual[name] != expected[name]} == {"R/CP_SIR.R"}


def test_regeneration_rejects_changed_or_added_source(tmp_path):
    original = b"double objective() { return 1; }\n"
    path = tmp_path / "model.cpp"
    path.write_bytes(original)
    expected = {"revision": VENDOR.SOURCE_REVISION, "files": {
        path.name: {"source_sha256": hashlib.sha256(original).hexdigest()}}}
    VENDOR.validate_source(tmp_path, expected)
    path.write_bytes(original.replace(b"return 1", b"return 2"))
    with pytest.raises(ValueError, match="Source differs"):
        VENDOR.validate_source(tmp_path, expected)
    path.write_bytes(original)
    (tmp_path / "extra.cpp").write_text("void extra() {}")
    with pytest.raises(ValueError, match="file list"):
        VENDOR.validate_source(tmp_path, expected)


def test_regeneration_rejects_a_different_revision(tmp_path):
    with pytest.raises(ValueError, match="revision"):
        VENDOR.validate_source(tmp_path, {"revision": "not-the-baseline", "files": {}})


def test_source_fingerprints_accept_checkout_line_endings_only(tmp_path):
    original = b"double objective() {\n    return 1;\n}\n"
    path = tmp_path / "model.cpp"
    expected = {"revision": VENDOR.SOURCE_REVISION, "files": {
        path.name: {"source_sha256": hashlib.sha256(original).hexdigest()}}}
    for content in (original, original.replace(b"\n", b"\r\n")):
        path.write_bytes(content)
        VENDOR.validate_source(tmp_path, expected)
        assert VENDOR.source_bytes(path) == original
        path.write_bytes(content.replace(b"return 1", b"return 2"))
        with pytest.raises(ValueError, match="Source differs"):
            VENDOR.validate_source(tmp_path, expected)
