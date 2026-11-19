#!/usr/bin/env python3
"""Mechanically adapt the verified orthoDr C++ kernels to the Python bridge.

Pass the R package's src directory explicitly. The source is never modified.
The adapter changes only interface types, diagnostics and includes. gen_solver
is implemented in Python because it invokes user callbacks. Source license
headers are retained; the rewritten pseudo solver's original GPL-2.0-or-later
notice is additionally preserved in LICENSES/orthoDr-source-notice.txt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

DESTINATION = Path(__file__).resolve().parents[1] / "src" / "cpp"
EXCLUDED = {"gen_solver.cpp", "orthoDr_gen.h", "RcppExports.cpp"}
SOURCE_REVISION = "a9588d8dd73c39188d1c46f371763addb10bee9a"
SOURCE_MANIFEST = Path(__file__).with_name("r_source_manifest.json")
NATIVE_OVERLAY = Path(__file__).with_name("r_native_source_overlay.json")


def source_bytes(path: Path) -> bytes:
    """Fingerprint text independently of Git's checkout line endings."""
    return path.read_bytes().replace(b"\r\n", b"\n")


def validate_source(source: Path, expected: dict) -> None:
    """Reject changed inputs before labeling generated code with the pinned revision."""
    if expected["revision"] != SOURCE_REVISION:
        raise ValueError("The recorded source revision does not match this adapter")
    files = {p.name: p for p in source.iterdir()
             if p.suffix in {".cpp", ".h"} and p.name not in EXCLUDED}
    if set(files) != set(expected["files"]):
        raise ValueError("Source file list differs from the pinned R revision; use a git archive of that revision")
    changed = [name for name, path in files.items()
               if hashlib.sha256(source_bytes(path)).hexdigest()
               != expected["files"][name]["source_sha256"]]
    if changed:
        raise ValueError("Source differs from the verified R fingerprints: " + ", ".join(sorted(changed)))


def transform(text: str, name: str) -> str:
    text = re.sub(r"#include <RcppArmadillo\.h>", '#include "orthodr_bridge.h"', text)
    text = re.sub(r"#include <Rcpp\.h>\s*\n", "", text)
    text = text.replace("using namespace Rcpp;", "using namespace orthodr_py;")
    text = text.replace("Rcpp::", "orthodr_py::")
    text = re.sub(r'_\["([^"\\]+)"\]', r'Named("\1")', text)
    text = re.sub(r"^\s*//\s*\[\[.*?\]\]\s*\n", "", text, flags=re.M)
    text = text.replace("orthodr_py::Rcout", "std::cout")
    text = re.sub(r"\bRcout\b", "std::cout", text)
    text = text.replace("Rprintf(", "orthodr_printf(")
    text = re.sub(r"\bNA_REAL\b", "orthodr_py::missing_value", text)
    text = re.sub(r"\bR_NilValue\b", "nullptr", text)
    text = re.sub(r"\bNumericMatrix\b", "arma::mat", text)
    text = text.replace("orthodr_py::arma::mat", "arma::mat")
    if name == "pdose_direct_solver.cpp":
        # Named numeric vectors are dictionaries in Python; unnamed bounds
        # remain numeric arrays. This is purely an output representation.
        start = text.index("orthodr_py::NumericVector direct_norms(")
        end = text.index("\n}", start) + 2
        block = text[start:end].replace("orthodr_py::NumericVector", "orthodr_py::List")
        text = text[:start] + block + text[end:]
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--dest", type=Path, default=DESTINATION)
    parser.add_argument("--native-overlay", "--regression-overlay", action="store_true",
                        help="use the fingerprinted native changes to the R baseline")
    args = parser.parse_args()
    source, dest = args.source.resolve(), args.dest.resolve()
    if not (source / "orthoDr_pdose.h").is_file():
        parser.error("source must be an orthoDr R-package src directory")
    if source == dest or source in dest.parents or dest in source.parents:
        parser.error("source and destination must not overlap")
    # The source pin is independent of generated outputs and old numerical
    # fixtures. A new revision must be verified before either is regenerated.
    pin = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    expected = {"revision": pin["source_sha"], "files": {
        Path(name).name: {"source_sha256": digest}
        for name, digest in pin["source_files_sha256"].items()
        if name.startswith("src/") and Path(name).suffix in {".cpp", ".h"}
        and Path(name).name not in EXCLUDED}}
    revision = SOURCE_REVISION
    overlay = None
    if args.native_overlay:
        overlay = json.loads(NATIVE_OVERLAY.read_text(encoding="utf-8"))
        if overlay["base_revision"] != SOURCE_REVISION:
            parser.error("Native overlay does not match the R baseline")
        overlay_files = overlay["files"]
        expected["files"].update(overlay_files)
        identity = hashlib.sha256(json.dumps(overlay_files, sort_keys=True).encode()).hexdigest()
        revision += "+native." + identity
    try:
        validate_source(source, expected)
    except ValueError as exc:
        parser.error(str(exc))
    manifest = {"upstream": "https://github.com/teazrq/orthoDr", "revision": revision,
                "license": "GPL-2.0-or-later",
                "source_hash_convention": pin["source_hash_convention"], "files": {}}
    if overlay:
        manifest["base_revision"] = SOURCE_REVISION
        manifest["source_overlay"] = sorted(overlay_files)
        for name, hashes in overlay_files.items():
            generated = transform(source_bytes(source / name).decode("utf-8"), name)
            if hashlib.sha256(generated.encode()).hexdigest() != hashes["generated_sha256"]:
                parser.error(f"Generated {name} differs from the recorded fingerprint")
    changed = 0
    for path in sorted(source.iterdir()):
        if path.suffix not in {".cpp", ".h"} or path.name in EXCLUDED:
            continue
        original = source_bytes(path)
        text = transform(original.decode("utf-8").replace("\r\n", "\n"), path.name)
        # Reject accidental runtime dependencies, ignoring comments.
        code = re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S)
        if re.search(r"\bRcpp\b|\bSEXP\b|R_NilValue|\bRf_", code):
            raise RuntimeError(f"Residual R dependency in {path.name}")
        target = dest / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.read_bytes() != text.encode("utf-8"):
            target.write_text(text, encoding="utf-8", newline="\n")
            changed += 1
        manifest["files"][path.name] = {
            "source_sha256": hashlib.sha256(original).hexdigest(),
            "generated_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
    (dest / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(manifest['files'])} files ({changed} changed); no R runtime dependency")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
