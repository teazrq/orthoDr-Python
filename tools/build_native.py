#!/usr/bin/env python3
"""Build on Windows with the existing shared Rtools compiler and libraries.

Uses the invoking Python installation and shared pybind11. No environment or
dependency is downloaded. Override ORTHODR_RTOOLS and ORTHODR_ARMADILLO_INCLUDE
to select other existing shared installations. Build artifacts and logs may be
placed outside the package with --build-dir.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import time

import pybind11

PACKAGE = Path(__file__).resolve().parents[1]


def build(output: Path, build_dir: Path, jobs: int = 3) -> dict:
    if os.name != "nt":
        raise RuntimeError("Use the standard setuptools build on Linux; this helper targets shared Windows Rtools")
    toolchain = Path(os.environ.get("ORTHODR_RTOOLS", "C:/rtools45/x86_64-w64-mingw32.static.posix"))
    compiler = toolchain / "bin/g++.exe"
    if not compiler.is_file():
        raise RuntimeError("Install Rtools globally or set ORTHODR_RTOOLS to the existing shared installation")
    configured_armadillo = os.environ.get("ORTHODR_ARMADILLO_INCLUDE")
    candidates = ([Path(configured_armadillo)] if configured_armadillo else []) + [
        Path("C:/Program Files/R/R-4.6.1/library/RcppArmadillo/include"), toolchain / "include"]
    armadillo = next((p for p in candidates if (p / "armadillo").is_file()), None)
    if armadillo is None:
        raise RuntimeError("Install Armadillo headers globally or set ORTHODR_ARMADILLO_INCLUDE")
    build_dir = build_dir.resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PATH"] = str(toolchain / "bin") + os.pathsep + environment.get("PATH", "")
    flags = ["-O2", "-std=c++17", "-fopenmp", "-DARMA_DONT_USE_WRAPPER", "-DARMA_WARN_LEVEL=1",
             "-DPYBIND11_DETAILED_ERROR_MESSAGES"]
    includes = [PACKAGE / "src", PACKAGE / "src/cpp", armadillo,
                Path(pybind11.get_include()), Path(sysconfig.get_path("include"))]
    flags += ["-I" + str(p) for p in includes]
    sources = sorted((PACKAGE / "src/cpp").glob("*.cpp")) + [PACKAGE / "src/bindings.cpp"]
    latest_header = max(p.stat().st_mtime for p in (PACKAGE / "src").rglob("*.h"))
    started = time.monotonic()

    def compile_one(source: Path) -> Path:
        name = source.relative_to(PACKAGE).as_posix().replace("/", "_")
        obj = build_dir / (name + ".o")
        command = [str(compiler), *flags, "-c", str(source), "-o", str(obj)]
        signature = build_dir / (name + ".command.json")
        encoded = json.dumps(command)
        if (obj.exists() and signature.exists() and signature.read_text() == encoded
                and obj.stat().st_mtime > max(source.stat().st_mtime, latest_header)):
            return obj
        result = subprocess.run(command, env=environment, capture_output=True, text=True)
        (build_dir / (name + ".log")).write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(name + "\n" + result.stderr[-7000:])
        signature.write_text(encoded)
        print("Compiled", source.relative_to(PACKAGE), flush=True)
        return obj

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        objects = list(pool.map(compile_one, sources))
    command = [str(compiler), "-shared", "-fopenmp", "-static-libgcc", "-static-libstdc++",
               *map(str, objects), "-L" + str(Path(sys.base_prefix) / "libs"),
               "-lpython" + str(sys.version_info.major) + str(sys.version_info.minor),
               "-llapack", "-lblas", "-lgfortran", "-lquadmath", "-o", str(output)]
    result = subprocess.run(command, env=environment, capture_output=True, text=True)
    (build_dir / "link.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    (build_dir / "link.command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    if result.returncode:
        raise RuntimeError(result.stderr[-7000:])
    metadata = {"python": sys.executable, "compiler": str(compiler), "armadillo_include": str(armadillo),
                "extension": str(output), "sources": len(sources), "seconds": time.monotonic() - started}
    (build_dir / "build.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=PACKAGE / "build/native")
    parser.add_argument("--output", type=Path, default=PACKAGE / "orthodr" / ("_core" + sysconfig.get_config_var("EXT_SUFFIX")))
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.build_dir, args.jobs), indent=2), flush=True)
