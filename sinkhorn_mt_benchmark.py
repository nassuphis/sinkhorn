from __future__ import annotations

import argparse
import ctypes
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from sinkhorn_benchmark import cost_matrix


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "sinkhorn_mt.c"


def library_path() -> Path:
    if sys.platform == "darwin":
        return ROOT / "libsinkhorn_mt.dylib"
    return ROOT / "libsinkhorn_mt.so"


def build_library() -> Path:
    lib = library_path()
    if sys.platform == "darwin":
        cmd = [
            "cc",
            "-O3",
            "-std=c11",
            "-pthread",
            "-fPIC",
            "-dynamiclib",
            str(SOURCE),
            "-o",
            str(lib),
        ]
    else:
        cmd = [
            "cc",
            "-O3",
            "-std=c11",
            "-pthread",
            "-fPIC",
            "-shared",
            str(SOURCE),
            "-o",
            str(lib),
            "-lm",
        ]

    subprocess.run(cmd, cwd=ROOT, check=True)
    return lib


class SinkhornMT:
    def __init__(self, lib_path: Path | None = None) -> None:
        self.lib_path = lib_path or library_path()
        self.lib = ctypes.CDLL(str(self.lib_path))
        self.lib.sinkhorn_mt_run.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        self.lib.sinkhorn_mt_run.restype = ctypes.c_int
        self.lib.hungarian_run.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
        ]
        self.lib.hungarian_run.restype = ctypes.c_int

    def run(
        self,
        cost: np.ndarray,
        epsilon: float,
        iterations: int,
        threads: int,
        return_plan: bool = False,
    ) -> dict[str, float | np.ndarray]:
        cost = np.ascontiguousarray(cost, dtype=np.float64)
        n, m = cost.shape
        if n != m:
            raise ValueError("cost matrix must be square")

        plan = np.empty_like(cost) if return_plan else None
        out_cost = ctypes.c_double()
        out_row_error = ctypes.c_double()
        out_col_error = ctypes.c_double()

        status = self.lib.sinkhorn_mt_run(
            cost.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(n),
            ctypes.c_double(epsilon),
            ctypes.c_int(iterations),
            ctypes.c_int(threads),
            (
                plan.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                if plan is not None
                else None
            ),
            ctypes.byref(out_cost),
            ctypes.byref(out_row_error),
            ctypes.byref(out_col_error),
        )
        if status != 0:
            raise RuntimeError(f"sinkhorn_mt_run failed with status {status}")

        result: dict[str, float | np.ndarray] = {
            "cost": out_cost.value,
            "row_error": out_row_error.value,
            "col_error": out_col_error.value,
        }
        if plan is not None:
            result["plan"] = plan
        return result

    def run_hungarian(
        self,
        cost: np.ndarray,
        return_perm: bool = False,
    ) -> dict[str, float | np.ndarray]:
        cost = np.ascontiguousarray(cost, dtype=np.float64)
        n, m = cost.shape
        if n != m:
            raise ValueError("cost matrix must be square")

        perm = np.empty(n, dtype=np.int32) if return_perm else None
        out_cost = ctypes.c_double()
        status = self.lib.hungarian_run(
            cost.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(n),
            (
                perm.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
                if perm is not None
                else None
            ),
            ctypes.byref(out_cost),
        )
        if status != 0:
            raise RuntimeError(f"hungarian_run failed with status {status}")

        result: dict[str, float | np.ndarray] = {"cost": out_cost.value}
        if perm is not None:
            result["perm"] = perm
        return result


def sorted_1d_cost(rng: np.random.Generator, n: int) -> np.ndarray:
    x = np.sort(rng.random(n))
    y = np.sort(rng.random(n))
    return cost_matrix(x, y)


def dense_random_cost(rng: np.random.Generator, n: int) -> np.ndarray:
    return np.ascontiguousarray(rng.random((n, n)), dtype=np.float64)


def flat_cost(n: int) -> np.ndarray:
    return np.zeros((n, n), dtype=np.float64)


def random_cost(rng: np.random.Generator, n: int, kind: str) -> np.ndarray:
    if kind == "sorted-1d":
        return sorted_1d_cost(rng, n)
    if kind == "dense":
        return dense_random_cost(rng, n)
    if kind == "flat":
        return flat_cost(n)
    raise ValueError(f"unknown cost kind: {kind}")


def time_mt(
    runner: SinkhornMT,
    cost: np.ndarray,
    epsilon: float,
    iterations: int,
    threads: int,
    repeats: int,
) -> tuple[float, dict[str, float | np.ndarray]]:
    best_ms = float("inf")
    best_result: dict[str, float | np.ndarray] | None = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = runner.run(cost, epsilon, iterations, threads)
        elapsed_ms = (time.perf_counter() - start) * 1_000.0
        if elapsed_ms < best_ms:
            best_ms = elapsed_ms
            best_result = result
    if best_result is None:
        raise RuntimeError("no benchmark result")
    return best_ms, best_result


def time_hungarian(
    runner: SinkhornMT,
    cost: np.ndarray,
    repeats: int,
) -> tuple[float, dict[str, float | np.ndarray]]:
    best_ms = float("inf")
    best_result: dict[str, float | np.ndarray] | None = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = runner.run_hungarian(cost)
        elapsed_ms = (time.perf_counter() - start) * 1_000.0
        if elapsed_ms < best_ms:
            best_ms = elapsed_ms
            best_result = result
    if best_result is None:
        raise RuntimeError("no Hungarian benchmark result")
    return best_ms, best_result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark pthread-parallel C Sinkhorn through ctypes."
    )
    parser.add_argument("--build", action="store_true", help="Build the C shared library.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 2000, 3000, 4000])
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--cost-kind",
        choices=["sorted-1d", "dense", "flat"],
        default="sorted-1d",
        help=(
            "sorted-1d uses squared distances between sorted random points; "
            "dense uses an unstructured random matrix; "
            "flat uses an all-zero matrix to expose worst-case tie behavior."
        ),
    )
    parser.add_argument(
        "--hungarian-size-limit",
        type=int,
        default=4000,
        help="Only time the serial C Hungarian baseline up to this n.",
    )
    args = parser.parse_args()

    if args.build or not library_path().exists():
        lib = build_library()
        print(f"built {lib}")

    runner = SinkhornMT()
    rng = np.random.default_rng(args.seed)
    print(f"platform = {platform.platform()}")
    print(f"epsilon = {args.epsilon}")
    print(f"iterations = {args.iterations}")
    print(f"cost_kind = {args.cost_kind}")
    print(f"repeats = {args.repeats} (best time reported)")
    print("hungarian baseline = serial C shortest-augmenting-path implementation")
    print()
    print(
        "n     hung_c_ms  threads  sink_c_ms speedup_vs_1t  "
        "speedup_vs_hung  sink_cost  hung_cost  row_err    col_err"
    )
    print("-" * 111)

    for n in args.sizes:
        cost = random_cost(rng, n, args.cost_kind)
        hungarian_ms: float | None = None
        hungarian_cost: float | None = None
        if n <= args.hungarian_size_limit:
            hungarian_ms, hungarian_result = time_hungarian(runner, cost, args.repeats)
            hungarian_cost = float(hungarian_result["cost"])

        one_thread_ms: float | None = None
        for threads in args.threads:
            sink_ms, result = time_mt(
                runner,
                cost,
                args.epsilon,
                args.iterations,
                threads,
                args.repeats,
            )
            if one_thread_ms is None:
                one_thread_ms = sink_ms

            speedup_vs_1t = one_thread_ms / sink_ms
            speedup_vs_hung = (
                hungarian_ms / sink_ms if hungarian_ms is not None else float("nan")
            )
            hungarian_label = f"{hungarian_ms:9.3f}" if hungarian_ms is not None else "        -"
            hungarian_cost_label = (
                f"{hungarian_cost:9.6f}" if hungarian_cost is not None else "        -"
            )
            speedup_hung_label = (
                f"{speedup_vs_hung:15.2f}"
                if hungarian_ms is not None
                else "              -"
            )
            print(
                f"{n:<5d} {hungarian_label}  "
                f"{threads:<7d} "
                f"{sink_ms:8.3f}  "
                f"{speedup_vs_1t:13.2f}  "
                f"{speedup_hung_label}  "
                f"{float(result['cost']):10.6f}  "
                f"{hungarian_cost_label}  "
                f"{float(result['row_error']):9.2e}  "
                f"{float(result['col_error']):9.2e}"
            )


if __name__ == "__main__":
    main()
