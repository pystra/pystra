"""Compare fixed reliability workloads with a separate PySTRA 1.6.0 checkout.

Run with the same interpreter/dependencies in both versions. Workers import only
one selected checkout. Timings exclude imports and model construction, include
analysis construction (1.x SORM/IS run FORM there), and retain fresh state per run.
Raw measurements, allocation peaks and profiles are written beside the summary.
"""

import argparse
import cProfile
import gc
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import pstats
import statistics
import subprocess
import sys
import time
import tracemalloc

CASES = (
    "form_normal",
    "form_nonnormal",
    "form_correlated",
    "sorm_nonlinear",
    "crude_mc",
    "importance_sampling",
)
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def make_problem(case, legacy):
    """Return matched model inputs and an independent evaluation counter."""
    import numpy as np
    import pystra as ra

    model = ra.StochasticModel()
    add = model.addVariable if legacy else model.add_variable
    if case == "form_normal":
        add(ra.Normal("R", 10.0, 2.0))
        add(ra.Normal("S", 5.0, 1.0))
    elif case == "sorm_nonlinear":
        add(ra.Lognormal("X1", 500.0, 100.0))
        add(ra.Normal("X2", 2000.0, 400.0))
        add(ra.Uniform("X3", 5.0, 0.5))
    else:
        add(ra.Lognormal("R", 10.0, 2.0))
        add(ra.Gumbel("S", 5.0, 1.0))
    if case in ("form_correlated", "crude_mc", "importance_sampling"):
        correlation = [[1.0, 0.4], [0.4, 1.0]]
    elif case == "sorm_nonlinear":
        correlation = [[1.0, 0.3, 0.2], [0.3, 1.0, 0.2], [0.2, 0.2, 1.0]]
    else:
        correlation = None
    if correlation is not None:
        setter = model.setCorrelation if legacy else model.set_correlation
        setter(ra.CorrelationMatrix(correlation))
    count = [0]

    def linear(R, S):
        count[0] += np.asarray(R).size
        return R - S

    def nonlinear(X1, X2, X3):
        count[0] += np.asarray(X1).size
        return 1.0 - X2 / (1000 * X3) - (X1 / (200 * X3)) ** 2

    return (
        model,
        ra.LimitState(nonlinear if case == "sorm_nonlinear" else linear),
        count,
    )


def make_analysis(case, legacy, model, response, samples, block_size):
    import pystra as ra

    if legacy:
        options = ra.AnalysisOptions()
        options.setPrintOutput(False)
        options.setSamples(samples)
        options.setBlockSize(block_size)
        options.target_cov = 0.0
        cls = {
            "sorm_nonlinear": ra.Sorm,
            "crude_mc": ra.CrudeMonteCarlo,
            "importance_sampling": ra.ImportanceSampling,
        }.get(case, ra.Form)
        return cls(
            stochastic_model=model, limit_state=response, analysis_options=options
        )
    if case in ("crude_mc", "importance_sampling"):
        cls = ra.CrudeMonteCarlo if case == "crude_mc" else ra.ImportanceSampling
        return cls(
            model,
            response,
            options=ra.SimulationOptions(
                n_samples=samples, block_size=block_size, target_cov=0.0
            ),
            rng=20260913,
        )
    form_options = ra.FORMOptions(block_size=block_size)
    if case == "sorm_nonlinear":
        return ra.SORM(model, response, options=ra.SORMOptions(form=form_options))
    return ra.FORM(model, response, options=form_options)


def execute(case, args, profile=None, memory=False):
    import numpy as np

    legacy = args.flavor == "baseline"
    model, response, count = make_problem(case, legacy)
    np.random.seed(20260913)
    gc.collect()
    if memory:
        tracemalloc.start()
    if profile is not None:
        profile.enable()
    start = time.perf_counter()
    analysis = make_analysis(
        case, legacy, model, response, args.samples, args.block_size
    )
    result = analysis.run()
    elapsed = time.perf_counter() - start
    if profile is not None:
        profile.disable()
    if memory:
        retained, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    else:
        retained = peak = None
    if legacy:
        beta = (
            analysis.betag_breitung if case == "sorm_nonlinear" else analysis.getBeta()
        )
        probability = (
            analysis.pf2_breitung if case == "sorm_nonlinear" else analysis.getFailure()
        )
    else:
        if hasattr(result, "converged") and not result.converged:
            raise RuntimeError(result.message)
        beta, probability = result.beta, result.failure_probability
    metrics = {
        "seconds": elapsed,
        "evaluations": count[0],
        "beta": float(np.asarray(beta).item()),
        "failure_probability": float(np.asarray(probability).item()),
    }
    if case in ("crude_mc", "importance_sampling"):
        prefix = "" if legacy else "_"
        completed = getattr(analysis, f"{prefix}k")
        assert completed == args.samples, (completed, args.samples)
        storage = {
            name: getattr(analysis, prefix + name).nbytes
            for name in ("u_all", "x_all", "all_G1", "approxMC_beta_all")
        }
        histories = (
            ("q_bar", "cov_q_bar") if legacy else ("q_bar", "cov_q_bar", "log_q_bar")
        )
        metrics.update(
            n_samples=completed,
            samples_per_second=completed / elapsed,
            stored_sample_bytes=sum(storage.values()),
            storage_arrays=storage,
            history_bytes=sum(getattr(analysis, prefix + n).nbytes for n in histories),
            coefficient_of_variation=float(
                getattr(analysis, prefix + "cov_q_bar")[completed - 1]
            ),
        )
    if memory:
        metrics.update(traced_retained_bytes=retained, traced_peak_bytes=peak)
    return metrics


def prepare(case, args):
    """Time Nataf correlation calibration and factorization through init_run."""
    model, response, _ = make_problem(case, args.flavor == "baseline")
    analysis = make_analysis(
        "form_normal",
        args.flavor == "baseline",
        model,
        response,
        args.samples,
        args.block_size,
    )
    gc.collect()
    start = time.perf_counter()
    analysis.init_run()
    return time.perf_counter() - start


def environment():
    import pystra

    cpu = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu": cpu,
        "affinity": (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
        "load_average": os.getloadavg() if hasattr(os, "getloadavg") else None,
        "threads": {key: os.environ.get(key) for key in THREADS},
        "pystra": pystra.__version__,
        "pystra_file": pystra.__file__,
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "pandas", "matplotlib")
        },
    }


def worker(args):
    import pystra

    assert Path(pystra.__file__).resolve().is_relative_to(args.checkout / "src")
    if args.cpu is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {args.cpu})
    records = {"environment": environment(), "cases": {}}
    for case in args.cases:
        # Warm both the transformation and complete workload before measuring.
        prepare(case, args)
        execute(case, args)
        if args.measure == "time":
            runs, preparation = [], []
            elapsed = 0.0
            while elapsed < args.minimum_seconds or len(runs) < 3:
                run = execute(case, args)
                runs.append(run)
                elapsed += run["seconds"]
            preparation = [prepare(case, args) for _ in range(25)]
            record = {"runs": runs, "preparation_seconds": preparation}
        else:
            profile = cProfile.Profile()
            profile_runs = 1 if case in ("crude_mc", "importance_sampling") else 25
            for _ in range(profile_runs):
                execute(case, args, profile=profile)
            stream = io.StringIO()
            stats = pstats.Stats(profile, stream=stream)
            functions = [
                {
                    "file": key[0],
                    "line": key[1],
                    "function": key[2],
                    "calls": value[1],
                    "self_seconds": value[2],
                    "cumulative_seconds": value[3],
                }
                for key, value in stats.stats.items()
            ]
            stats.strip_dirs().sort_stats("cumulative").print_stats(35)
            (args.output.parent / f"{args.flavor}-{case}-profile.txt").write_text(
                stream.getvalue()
            )
            memory_runs = [execute(case, args, memory=True) for _ in range(3)]
            memory = dict(memory_runs[0])
            for name in ("traced_peak_bytes", "traced_retained_bytes"):
                memory[name] = statistics.median(run[name] for run in memory_runs)
            record = {
                "memory": memory,
                "memory_runs": memory_runs,
                "profile_runs": profile_runs,
                "profile_functions": sorted(
                    functions, key=lambda row: row["cumulative_seconds"], reverse=True
                ),
            }
        records["cases"][case] = record
    args.output.write_text(json.dumps(records, indent=2) + "\n")


def describe(values):
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return {
        "median": statistics.median(values),
        "q1": quartiles[0],
        "q3": quartiles[2],
        "values": values,
    }


def compare(args, rounds, details):
    result = {"cases": {}, "flags": []}
    for case in args.cases:
        case_result = {}
        for flavor in ("baseline", "current"):
            values = [r[flavor]["cases"][case] for r in rounds]
            run_times = [
                statistics.median(v["seconds"] for v in r["runs"]) for r in values
            ]
            preparation = [statistics.median(r["preparation_seconds"]) for r in values]
            case_result[flavor] = {
                "run_seconds": describe(run_times),
                "preparation_seconds": describe(preparation),
                "evaluations": sorted(
                    {v["evaluations"] for r in values for v in r["runs"]}
                ),
                "reference": values[0]["runs"][0],
                "memory": details[flavor]["cases"][case]["memory"],
                "memory_runs": details[flavor]["cases"][case].get("memory_runs", []),
            }
        for metric in ("run_seconds", "preparation_seconds"):
            baseline = case_result["baseline"][metric]["median"]
            current = case_result["current"][metric]["median"]
            increase = 100 * (current / baseline - 1)
            case_result[f"{metric}_change_percent"] = increase
            if increase > 10:
                result["flags"].append(
                    {"case": case, "metric": metric, "increase_percent": increase}
                )
        for metric in ("traced_peak_bytes", "stored_sample_bytes", "history_bytes"):
            baseline = case_result["baseline"]["memory"].get(metric)
            current = case_result["current"]["memory"].get(metric)
            if baseline:
                increase = 100 * (current / baseline - 1)
                case_result[f"{metric}_change_percent"] = increase
                if increase > 10:
                    result["flags"].append(
                        {"case": case, "metric": metric, "increase_percent": increase}
                    )
        if (
            case_result["current"]["evaluations"]
            != case_result["baseline"]["evaluations"]
        ):
            result["flags"].append(
                {
                    "case": case,
                    "metric": "evaluations",
                    "baseline": case_result["baseline"]["evaluations"],
                    "current": case_result["current"]["evaluations"],
                }
            )
        baseline = case_result["baseline"]["reference"]
        current = case_result["current"]["reference"]
        if case in ("crude_mc", "importance_sampling"):
            import math

            standard_error = math.hypot(
                baseline["failure_probability"] * baseline["coefficient_of_variation"],
                current["failure_probability"] * current["coefficient_of_variation"],
            )
            difference = abs(
                current["failure_probability"] - baseline["failure_probability"]
            )
            case_result["numerical_comparison"] = {
                "pooled_standard_errors": difference / standard_error,
                "passed": difference <= 5 * standard_error,
            }
            for flavor in ("baseline", "current"):
                case_result[flavor]["sampling_throughput"] = (
                    args.samples / case_result[flavor]["run_seconds"]["median"]
                )
        else:
            tolerance = 2e-4 if case == "sorm_nonlinear" else 1e-10
            difference = abs(current["beta"] - baseline["beta"])
            case_result["numerical_comparison"] = {
                "beta_absolute_difference": difference,
                "tolerance": tolerance,
                "passed": difference <= tolerance,
            }
        result["cases"][case] = case_result
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument(
        "--current", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--block-size", type=int, default=1000)
    parser.add_argument("--minimum-seconds", type=float, default=0.10)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Recompute the summary from existing raw measurements",
    )
    parser.add_argument("--checkout", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--flavor", choices=("baseline", "current"), help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--measure", choices=("time", "detail"), default="time", help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if (
        args.baseline is None
        or args.rounds < 3
        or args.samples < 2
        or args.block_size < 1
        or args.minimum_seconds <= 0
    ):
        parser.error(
            "provide --baseline, at least three rounds, positive durations/block size and at least two samples"
        )
    args.output = args.output.resolve()
    if args.summarize:
        report = json.loads((args.output / "results.json").read_text())
        args.rounds = report["configuration"]["rounds"]
        args.samples = report["configuration"]["samples"]
        args.cases = list(report["cases"])
        rounds = [
            {
                flavor: json.loads((args.output / f"{index}-{flavor}.json").read_text())
                for flavor in ("baseline", "current")
            }
            for index in range(args.rounds)
        ]
        details = {
            flavor: json.loads((args.output / f"detail-{flavor}.json").read_text())
            for flavor in ("baseline", "current")
        }
        report.update(compare(args, rounds, details))
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        if not all(
            case["numerical_comparison"]["passed"] for case in report["cases"].values()
        ):
            raise SystemExit(1)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve()
    revisions = {}
    for flavor, checkout in (("baseline", args.baseline), ("current", args.current)):
        checkout = checkout.resolve()
        if not (checkout / "src/pystra/__init__.py").is_file():
            parser.error(f"not a PySTRA checkout: {checkout}")
        source_hash = hashlib.sha256()
        for path in sorted((checkout / "src/pystra").rglob("*.py")):
            source_hash.update(str(path.relative_to(checkout)).encode())
            source_hash.update(path.read_bytes())
        revision = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
        )
        revisions[flavor] = {
            "path": str(checkout),
            "revision": (
                revision.stdout.strip()
                if revision.returncode == 0
                else "export (see source hash)"
            ),
            "source_sha256": source_hash.hexdigest(),
        }

    def launch(flavor, index, measure):
        checkout = (args.baseline if flavor == "baseline" else args.current).resolve()
        output = args.output / f"{index}-{flavor}.json"
        command = [
            sys.executable,
            str(script),
            "--worker",
            "--checkout",
            str(checkout),
            "--flavor",
            flavor,
            "--output",
            str(output),
            "--measure",
            measure,
            "--samples",
            str(args.samples),
            "--block-size",
            str(args.block_size),
            "--minimum-seconds",
            str(args.minimum_seconds),
            "--cases",
            *args.cases,
        ]
        if args.cpu is not None:
            command += ["--cpu", str(args.cpu)]
        env = {
            **os.environ,
            **{key: "1" for key in THREADS},
            "PYTHONPATH": str(checkout / "src"),
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(args.output / "matplotlib"),
        }
        completed = subprocess.run(
            command, env=env, cwd=args.output, text=True, capture_output=True
        )
        (args.output / f"{index}-{flavor}.log").write_text(
            completed.stdout + completed.stderr
        )
        completed.check_returncode()
        data = json.loads(output.read_text())
        print(f"Completed {index}: {flavor}", flush=True)
        return data

    rounds = []
    for index in range(args.rounds):
        order = ("baseline", "current") if index % 2 == 0 else ("current", "baseline")
        rounds.append({flavor: launch(flavor, index, "time") for flavor in order})
    details = {
        flavor: launch(flavor, "detail", "detail") for flavor in ("baseline", "current")
    }
    report = {
        "schema_version": 1,
        "revisions": revisions,
        "configuration": {
            "rounds": args.rounds,
            "samples": args.samples,
            "block_size": args.block_size,
            "minimum_seconds": args.minimum_seconds,
            "cpu": args.cpu,
        },
        "environments": {key: value["environment"] for key, value in details.items()},
        **compare(args, rounds, details),
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["flags"], indent=2))
    print(f"Results: {args.output / 'results.json'}")
    if not all(
        case["numerical_comparison"]["passed"] for case in report["cases"].values()
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
