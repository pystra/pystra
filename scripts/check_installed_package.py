"""Build distributions and exercise an installed wheel in a fresh external venv.

The default uses the package index. Supply --wheelhouse for offline dependencies,
or --uv-cache-source to reconstruct compatible wheels from verified uv cache
archives. Examples are copied byte-for-byte and run with isolated Python from an
external directory. Every attempted command and its outcome is retained.
"""

import argparse
import base64
import csv
from email import message_from_bytes
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
from zipfile import ZIP_DEFLATED, ZipFile

THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def cached_wheelhouse(cache: Path, output: Path) -> list[dict]:
    """Repack compatible cache archives after checking every RECORD digest.

    Only cached PyPI distributions in the dependency closure are considered;
    this does not copy an installed environment or expose system site-packages.
    The archive's METADATA and wheel tags drive selection and dependencies.
    """
    from packaging.requirements import Requirement
    from packaging.tags import sys_tags
    from packaging.utils import canonicalize_name, parse_wheel_filename

    tags = set(sys_tags())
    pending = ["setuptools", "numpy", "scipy", "pandas", "matplotlib"]
    seen = set()
    records = []
    output.mkdir(parents=True, exist_ok=True)
    while pending:
        name = canonicalize_name(pending.pop())
        if name in seen:
            continue
        seen.add(name)
        candidates = []
        for path in (cache / "wheels-v6/pypi" / name).glob("*"):
            if not path.is_dir():
                continue
            filename = name.replace("-", "_") + "-" + path.name + ".whl"
            _, version, _, wheel_tags = parse_wheel_filename(filename)
            if tags.intersection(wheel_tags):
                candidates.append((version, filename, path))
        if not candidates:
            raise RuntimeError(f"No compatible cached wheel for {name}")
        _, filename, archive = max(candidates)
        metadata_path = next(archive.glob("*.dist-info/METADATA"))
        metadata = message_from_bytes(metadata_path.read_bytes())
        record_path = metadata_path.with_name("RECORD")
        entries = list(csv.reader(io.StringIO(record_path.read_text())))
        for relative, digest, size in entries:
            path = (archive / relative).resolve()
            if not path.is_relative_to(archive.resolve()):
                raise ValueError(f"RECORD leaves cached wheel: {relative}")
            content = path.read_bytes()
            if digest:
                algorithm, expected = digest.split("=", 1)
                actual = (
                    base64.urlsafe_b64encode(hashlib.new(algorithm, content).digest())
                    .rstrip(b"=")
                    .decode()
                )
                if actual != expected or len(content) != int(size):
                    raise ValueError(f"Cached wheel RECORD mismatch: {path}")
        destination = output / filename
        with ZipFile(destination, "w", compression=ZIP_DEFLATED) as wheel:
            for relative, _, _ in entries:
                wheel.write(archive / relative, relative)
        records.append(
            {
                "name": name,
                "version": metadata["Version"],
                "cache_archive": str(archive.resolve()),
                "wheel": filename,
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "verified_record_entries": len(entries),
            }
        )
        for entry in metadata.get_all("Requires-Dist", []):
            requirement = Requirement(entry)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                pending.append(requirement.name)
    return records


def installed_smoke(output: Path):
    """Check provenance, public imports and an analytic reference after install."""
    import importlib
    import site
    import numpy as np
    from scipy.stats import norm
    import pystra as ra
    from pystra.migrate import convert_source

    assert sys.prefix != sys.base_prefix
    assert not site.ENABLE_USER_SITE
    assert Path(ra.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    exports = {name: type(getattr(ra, name)).__name__ for name in ra.__all__}
    modules = [
        "pystra.model",
        "pystra.distributions",
        "pystra.dependence",
        "pystra.reliability",
        "pystra.calibration",
        "pystra.decision",
        "pystra.active_learning",
        "pystra.results",
        "pystra.migrate",
    ]
    for name in modules:
        importlib.import_module(name)
    conversion = convert_source(
        "import pystra as ra\nmodel = ra.Form(stochastic_model=m, limit_state=g)\n"
    )
    assert "ra.FORM(model=m, limit_state=g)" in conversion.source
    model = ra.StochasticModel()
    model.add_variable(ra.Normal("R", 10.0, 2.0))
    model.add_variable(ra.Normal("S", 5.0, 1.0))
    response = ra.LimitState(lambda R, S: R - S)
    form = ra.FORM(model, response)
    form_result = form.run()
    sorm_result = ra.SORM(model, response, form=form).run()
    simulation = ra.CrudeMonteCarlo(
        model,
        response,
        options=ra.SimulationOptions(n_samples=20_000, target_cov=0),
        rng=20260913,
    ).run()
    reference = norm.sf(np.sqrt(5.0))
    assert form_result.converged and sorm_result.converged
    np.testing.assert_allclose(form_result.beta, np.sqrt(5.0), rtol=1e-9)
    np.testing.assert_allclose(sorm_result.failure_probability, reference, rtol=1e-8)
    standard_error = np.sqrt(reference * (1 - reference) / simulation.n_samples)
    assert abs(simulation.failure_probability - reference) <= 5 * standard_error
    assert simulation.n_samples == 20_000
    data = {
        "python": sys.version,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "cwd": str(Path.cwd()),
        "sys_path": sys.path,
        "pystra_file": ra.__file__,
        "version": ra.__version__,
        "exports": exports,
        "modules": modules,
        "dependencies": {
            dist.metadata["Name"]: dist.version
            for dist in importlib.metadata.distributions()
        },
        "form_beta": form_result.beta,
        "sorm_failure_probability": sorm_result.failure_probability,
        "mc_failure_probability": simulation.failure_probability,
        "reference_probability": float(reference),
        "mc_reference_standard_errors": float(
            abs(simulation.failure_probability - reference) / standard_error
        ),
    }
    output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path)
    parser.add_argument("--uv-cache-source", type=Path)
    parser.add_argument(
        "--with-opensees",
        action="store_true",
        help="Install the optional OpenSees dependency before running its example",
    )
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument(
        "--installed-smoke", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.installed_smoke:
        installed_smoke(args.output)
        return
    source, output = args.source.resolve(), args.output.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        parser.error("--output must be outside the source checkout and its ancestors")
    if args.wheelhouse and args.uv_cache_source:
        parser.error("choose --wheelhouse or --uv-cache-source")
    output.mkdir(parents=True, exist_ok=False)
    runtime = output / "run"
    runtime.mkdir()
    logs = output / "logs"
    logs.mkdir()
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")
    }
    environment.update(
        {
            **{key: "1" for key in THREADS},
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(output / "matplotlib"),
            "UV_CACHE_DIR": str(output / "uv-cache"),
            "UV_PYTHON_DOWNLOADS": "never",
        }
    )
    source_hash = hashlib.sha256()
    for path in sorted((source / "src/pystra").rglob("*.py")):
        source_hash.update(str(path.relative_to(source)).encode())
        source_hash.update(path.read_bytes())
    records = []
    report = {
        "source": str(source),
        "source_sha256": source_hash.hexdigest(),
        "source_revision": subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "commands": records,
        "environment": {
            key: environment[key] for key in (*THREADS, "MPLBACKEND", "MPLCONFIGDIR")
        },
    }

    def save():
        # Commands execute with their real paths; only exported evidence is portable.
        roots = {
            str(source): "<repo>",
            str(output / "venv"): "<venv>",
            str(output): "<output>",
            sys.executable: "<python>",
            sys.prefix: "<python-prefix>",
            str(Path.home()): "<home>",
            shutil.which("uv"): "<uv>",
        }
        if args.uv_cache_source:
            roots[str(args.uv_cache_source.resolve())] = "<cache>"
        if args.wheelhouse:
            roots[str(args.wheelhouse.resolve())] = "<wheelhouse>"
        text = json.dumps(report, indent=2)
        for root in sorted((root for root in roots if root), key=len, reverse=True):
            text = text.replace(json.dumps(root)[1:-1], roots[root])
        (output / "results.json").write_text(text + "\n")

    def run(label, command, *, required=True, optional=False):
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=runtime,
                env=environment,
                text=True,
                capture_output=True,
                timeout=args.timeout,
            )
            code, stdout, stderr = (
                completed.returncode,
                completed.stdout,
                completed.stderr,
            )
        except subprocess.TimeoutExpired as error:
            code = 124
            stdout = error.stdout or b""
            stderr = (error.stderr or b"") + b"\nTimed out\n"
            stdout = (
                stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout
            )
            stderr = (
                stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
            )
        logfile = logs / f"{label}.log"
        logfile.write_text(stdout + stderr)
        records.append(
            {
                "label": label,
                "command": command,
                "cwd": str(runtime),
                "returncode": code,
                "seconds": time.perf_counter() - started,
                "optional": optional,
                "log": str(logfile),
            }
        )
        save()
        print(f"{label}: exit {code}", flush=True)
        if code and required:
            raise RuntimeError(f"{label} failed; see {logfile}")
        return code

    wheelhouse = args.wheelhouse.resolve() if args.wheelhouse else None
    if args.uv_cache_source:
        wheelhouse = output / "dependency-wheels"
        report["cached_dependencies"] = cached_wheelhouse(
            args.uv_cache_source.resolve(), wheelhouse
        )
        save()
    index_options = (
        ["--offline", "--no-index", "--find-links", str(wheelhouse)]
        if wheelhouse
        else []
    )
    uv = shutil.which("uv")
    if uv is None:
        parser.error("uv is required")
    run("uv-version", [uv, "--version"])
    artifacts = output / "dist"
    run(
        "build",
        [
            uv,
            "build",
            "--python",
            sys.executable,
            "--out-dir",
            str(artifacts),
            *index_options,
            str(source),
        ],
    )
    wheel = next(artifacts.glob("*.whl"))
    sdist = next(artifacts.glob("*.tar.gz"))
    with ZipFile(wheel) as archive:
        assert archive.testzip() is None
        assert "pystra/migrate/_data.py" in archive.namelist()
        metadata = message_from_bytes(
            archive.read(
                next(
                    name
                    for name in archive.namelist()
                    if name.endswith(".dist-info/METADATA")
                )
            )
        )
        report["metadata"] = {
            key: metadata.get_all(key)
            for key in (
                "Name",
                "Version",
                "Requires-Python",
                "Requires-Dist",
                "Description-Content-Type",
                "License-File",
            )
        }
    with tarfile.open(sdist) as archive:
        names = archive.getnames()
        assert any(name.endswith("/src/pystra/migrate/_data.py") for name in names)
        assert not any(
            "/.codex-commits/" in name or "/.claude/" in name for name in names
        )
    report["artifacts"] = {
        p.name: {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "bytes": p.stat().st_size,
        }
        for p in (wheel, sdist)
    }
    save()
    venv = output / "venv"
    run("venv", [uv, "venv", "--python", sys.executable, str(venv)])
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    run(
        "install",
        [uv, "pip", "install", "--python", str(python), *index_options, str(wheel)],
    )
    run("dependency-check", [uv, "pip", "check", "--python", str(python)])
    run("freeze", [uv, "pip", "freeze", "--python", str(python)])
    copied_checker = runtime / "check_installed_package.py"
    shutil.copyfile(__file__, copied_checker)
    run(
        "imports-and-session",
        [
            str(python),
            "-I",
            str(copied_checker),
            "--installed-smoke",
            "--output",
            str(output / "installed-session.json"),
        ],
    )
    report["session"] = json.loads((output / "installed-session.json").read_text())
    run("converter-help", [str(python), "-I", "-m", "pystra.migrate", "--help"])
    if args.with_opensees:
        run(
            "optional-opensees-install",
            [
                uv,
                "pip",
                "install",
                "--python",
                str(python),
                *index_options,
                "openseespy",
            ],
            required=False,
            optional=True,
        )
    example_records = []
    for example in sorted((source / "examples").glob("*.py")):
        copied = runtime / example.name
        shutil.copyfile(example, copied)
        digest = hashlib.sha256(copied.read_bytes()).hexdigest()
        assert copied.read_bytes() == example.read_bytes()
        optional = example.name == "openseespy_ex.py"
        code = run(
            example.stem,
            [str(python), "-I", str(copied)],
            required=False,
            optional=optional,
        )
        example_records.append(
            {
                "name": example.name,
                "sha256": digest,
                "returncode": code,
                "optional": optional,
            }
        )
    report["examples"] = example_records
    report["core_passed"] = all(
        r["returncode"] == 0 for r in records if not r["optional"]
    )
    report["optional_passed"] = all(
        r["returncode"] == 0 for r in records if r["optional"]
    )
    save()
    raise SystemExit(0 if report["core_passed"] else 1)


if __name__ == "__main__":
    main()
