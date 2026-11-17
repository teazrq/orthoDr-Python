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
    generated = json.loads((ROOT / "src/cpp/provenance.json").read_text())
    assert pin["source_sha"] == VENDOR.SOURCE_REVISION == generated["revision"] == _core.source_revision
    assert pin["source_hash_convention"] == generated["source_hash_convention"] == (
        "CRLF normalized to LF; all other bytes unchanged")
    assert set(pin["source_files_md5"]) == set(pin["source_files_sha256"])
    for name, hashes in generated["files"].items():
        assert hashes["source_sha256"] == pin["source_files_sha256"]["src/" + name]
    for name in ("r_reference.json", "r_extended.json", "r_pseudo.json"):
        provenance = json.loads((ROOT / "tests/fixtures" / name).read_text())["provenance"]
        assert provenance["source_sha"] == pin["source_sha"], name
        assert provenance["source_files_md5"] == pin["source_files_md5"], name
        assert provenance["source_hash_convention"] == pin["source_hash_convention"], name


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
