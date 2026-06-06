from __future__ import annotations

import argparse
import itertools
import math
import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExactResult:
    perm: tuple[int, ...]
    plan: np.ndarray
    cost: float


@dataclass(frozen=True)
class SinkhornResult:
    plan: np.ndarray
    cost: float
    entropy: float
    iterations: int
    row_error: float
    col_error: float


@dataclass(frozen=True)
class TimedSinkhornResult:
    result: SinkhornResult
    elapsed_ms: float


def cost_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (x[:, None] - y[None, :]) ** 2


def logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    shifted = np.exp(values - max_values)
    result = max_values + np.log(np.sum(shifted, axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def entropy(plan: np.ndarray) -> float:
    positive = plan[plan > 0.0]
    return float(-np.sum(positive * np.log(positive)))


def assignment_overlap_and_error(exact: ExactResult, soft_plan: np.ndarray) -> tuple[float, float]:
    """Compare hard and soft assignments after scaling rows/cols to sum to 1.

    `exact.plan` and `soft_plan` are transport plans with total mass 1 and
    row/column sums 1/n. Multiplying by n gives assignment matrices with
    row/column sums 1. The normalized L1 error is in [0, 1].
    """
    n = soft_plan.shape[0]
    rows = np.arange(n)
    overlap = float(soft_plan[rows, exact.perm].sum())
    hard_assignment = exact.plan * n
    soft_assignment = soft_plan * n
    normalized_l1 = float(np.sum(np.abs(hard_assignment - soft_assignment)) / (2.0 * n))
    return overlap, normalized_l1


def brute_force_assignment(cost: np.ndarray) -> ExactResult:
    """Brute-force the tiny square assignment problem.

    This is the exact same optimum a Hungarian solver would return, but it keeps
    a readable sanity check around for very small n.
    """
    n, m = cost.shape
    if n != m:
        raise ValueError("exact_assignment expects a square cost matrix")

    rows = np.arange(n)
    best_perm: tuple[int, ...] | None = None
    best_sum = math.inf

    for perm in itertools.permutations(range(n)):
        total = float(cost[rows, perm].sum())
        if total < best_sum:
            best_sum = total
            best_perm = perm

    if best_perm is None:
        raise RuntimeError("no assignment found")

    plan = np.zeros_like(cost, dtype=float)
    plan[rows, np.array(best_perm)] = 1.0 / n
    return ExactResult(perm=best_perm, plan=plan, cost=float(np.sum(plan * cost)))


def exact_assignment(cost: np.ndarray) -> ExactResult:
    """Solve the square assignment problem with a dependency-free Hungarian core.

    This is the standard O(n^3) shortest-augmenting-path form of the Hungarian
    algorithm for minimum-cost bipartite matching. It returns the same hard
    one-to-one optimum as the brute-force baseline, but can handle n=200.
    """
    n, m = cost.shape
    if n != m:
        raise ValueError("exact_assignment expects a square cost matrix")

    # One-indexed arrays mirror the compact textbook implementation.
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)
    way = np.zeros(m + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, math.inf)
        used = np.zeros(m + 1, dtype=bool)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = math.inf
            j1 = 0

            for j in range(1, m + 1):
                if used[j]:
                    continue
                current = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if current < minv[j]:
                    minv[j] = current
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j

            for j in range(0, m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    perm = np.empty(n, dtype=int)
    for j in range(1, m + 1):
        if p[j] != 0:
            perm[p[j] - 1] = j - 1

    rows = np.arange(n)
    plan = np.zeros_like(cost, dtype=float)
    plan[rows, perm] = 1.0 / n
    return ExactResult(
        perm=tuple(int(j) for j in perm),
        plan=plan,
        cost=float(np.sum(plan * cost)),
    )


def sinkhorn(
    cost: np.ndarray,
    epsilon: float,
    max_iter: int = 2_000,
    tol: float = 1e-10,
) -> SinkhornResult:
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    n, m = cost.shape
    a = np.ones(n) / n
    b = np.ones(m) / m
    kernel = np.exp(-cost / epsilon)

    u = np.ones(n)
    v = np.ones(m)
    tiny = np.finfo(float).tiny

    row_error = math.inf
    col_error = math.inf
    iterations = max_iter

    for step in range(1, max_iter + 1):
        u = a / np.maximum(kernel @ v, tiny)
        v = b / np.maximum(kernel.T @ u, tiny)

        if step % 10 == 0 or step == max_iter:
            plan = (u[:, None] * kernel) * v[None, :]
            row_error = float(np.max(np.abs(plan.sum(axis=1) - a)))
            col_error = float(np.max(np.abs(plan.sum(axis=0) - b)))
            if max(row_error, col_error) < tol:
                iterations = step
                break

    plan = (u[:, None] * kernel) * v[None, :]
    return SinkhornResult(
        plan=plan,
        cost=float(np.sum(plan * cost)),
        entropy=entropy(plan),
        iterations=iterations,
        row_error=float(np.max(np.abs(plan.sum(axis=1) - a))),
        col_error=float(np.max(np.abs(plan.sum(axis=0) - b))),
    )


def log_sinkhorn(
    cost: np.ndarray,
    epsilon: float,
    max_iter: int = 20_000,
    tol: float = 1e-10,
) -> SinkhornResult:
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    n, m = cost.shape
    log_a = np.full(n, -math.log(n))
    log_b = np.full(m, -math.log(m))
    log_kernel = -cost / epsilon

    log_u = np.zeros(n)
    log_v = np.zeros(m)
    iterations = max_iter

    for step in range(1, max_iter + 1):
        log_u = log_a - logsumexp(log_kernel + log_v[None, :], axis=1)
        log_v = log_b - logsumexp(log_kernel + log_u[:, None], axis=0)

        if step % 10 == 0 or step == max_iter:
            log_plan = log_u[:, None] + log_kernel + log_v[None, :]
            plan = np.exp(log_plan)
            row_error = float(np.max(np.abs(plan.sum(axis=1) - np.exp(log_a))))
            col_error = float(np.max(np.abs(plan.sum(axis=0) - np.exp(log_b))))
            if max(row_error, col_error) < tol:
                iterations = step
                break

    log_plan = log_u[:, None] + log_kernel + log_v[None, :]
    plan = np.exp(log_plan)
    a = np.exp(log_a)
    b = np.exp(log_b)
    return SinkhornResult(
        plan=plan,
        cost=float(np.sum(plan * cost)),
        entropy=entropy(plan),
        iterations=iterations,
        row_error=float(np.max(np.abs(plan.sum(axis=1) - a))),
        col_error=float(np.max(np.abs(plan.sum(axis=0) - b))),
    )


def log_sinkhorn_from_duals(
    cost: np.ndarray,
    epsilon: float,
    f: np.ndarray,
    g: np.ndarray,
    max_iter: int,
    tol: float,
) -> tuple[SinkhornResult, np.ndarray, np.ndarray]:
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    n, m = cost.shape
    log_a = np.full(n, -math.log(n))
    log_b = np.full(m, -math.log(m))
    iterations = max_iter

    for step in range(1, max_iter + 1):
        f = epsilon * (log_a - logsumexp((g[None, :] - cost) / epsilon, axis=1))
        g = epsilon * (log_b - logsumexp((f[:, None] - cost) / epsilon, axis=0))

        if step % 100 == 0 or step == max_iter:
            plan = np.exp((f[:, None] + g[None, :] - cost) / epsilon)
            row_error = float(np.max(np.abs(plan.sum(axis=1) - np.exp(log_a))))
            col_error = float(np.max(np.abs(plan.sum(axis=0) - np.exp(log_b))))
            if max(row_error, col_error) < tol:
                iterations = step
                break

    plan = np.exp((f[:, None] + g[None, :] - cost) / epsilon)
    a = np.exp(log_a)
    b = np.exp(log_b)
    result = SinkhornResult(
        plan=plan,
        cost=float(np.sum(plan * cost)),
        entropy=entropy(plan),
        iterations=iterations,
        row_error=float(np.max(np.abs(plan.sum(axis=1) - a))),
        col_error=float(np.max(np.abs(plan.sum(axis=0) - b))),
    )
    return result, f, g


def sinkhorn_for_epsilons(
    cost: np.ndarray,
    epsilons: list[float],
    method: str,
    max_iter: int,
    tol: float,
) -> dict[float, TimedSinkhornResult]:
    if method in {"primal", "log"}:
        sinkhorn_fn = log_sinkhorn if method == "log" else sinkhorn
        return {
            epsilon: TimedSinkhornResult(*timed_call(
                sinkhorn_fn,
                cost,
                epsilon,
                max_iter=max_iter,
                tol=tol,
            ))
            for epsilon in epsilons
        }

    if method != "scaling":
        raise ValueError(f"unknown method: {method}")

    n, m = cost.shape
    f = np.zeros(n)
    g = np.zeros(m)
    results: dict[float, TimedSinkhornResult] = {}
    for epsilon in sorted(epsilons, reverse=True):
        start = time.perf_counter()
        result, f, g = log_sinkhorn_from_duals(
            cost,
            epsilon,
            f,
            g,
            max_iter=max_iter,
            tol=tol,
        )
        elapsed_ms = (time.perf_counter() - start) * 1_000.0
        results[epsilon] = TimedSinkhornResult(result=result, elapsed_ms=elapsed_ms)
    return results


def random_points(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(rng.random(n))
    y = np.sort(rng.random(n))
    return x, y


def format_array(array: np.ndarray, precision: int = 3) -> str:
    return np.array2string(
        array,
        precision=precision,
        suppress_small=True,
        floatmode="fixed",
    )


def timed_call(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1_000.0
    return result, elapsed_ms


def print_detail(
    rng: np.random.Generator,
    n: int,
    epsilons: list[float],
    max_iter: int,
    tol: float,
    method: str,
) -> None:
    x, y = random_points(rng, n)
    cost = cost_matrix(x, y)
    exact, exact_ms = timed_call(exact_assignment, cost)

    print(f"Detailed random example, n={n}")
    print(f"x = {format_array(x)}")
    print(f"y = {format_array(y)}")
    print("\nCost matrix:")
    print(format_array(cost))
    print("\nExact assignment baseline (Hungarian algorithm):")
    print(f"  permutation: {exact.perm}")
    print(f"  raw transport cost: {exact.cost:.8f}")
    print(f"  entropy: {entropy(exact.plan):.8f}")
    print(f"  time: {exact_ms:.3f} ms")
    print("  plan:")
    print(format_array(exact.plan, precision=4))

    sinkhorn_results = sinkhorn_for_epsilons(cost, epsilons, method, max_iter, tol)
    for epsilon in epsilons:
        timed_result = sinkhorn_results[epsilon]
        result = timed_result.result
        elapsed_ms = timed_result.elapsed_ms
        assignment_overlap, assignment_error = assignment_overlap_and_error(
            exact,
            result.plan,
        )
        print(f"\nSinkhorn epsilon={epsilon:g}:")
        print(f"  raw transport cost: {result.cost:.8f}")
        print(f"  cost gap vs exact: {result.cost - exact.cost:.8f}")
        print(f"  entropy: {result.entropy:.8f}")
        print(f"  assignment overlap: {assignment_overlap:.8f}")
        print(f"  assignment error: {assignment_error:.8f}")
        print(f"  iterations: {result.iterations}")
        print(f"  max row error: {result.row_error:.2e}")
        print(f"  max col error: {result.col_error:.2e}")
        print(f"  time: {elapsed_ms:.3f} ms")
        print("  plan:")
        print(format_array(result.plan, precision=4))


def print_benchmark(
    rng: np.random.Generator,
    sizes: list[int],
    trials: int,
    epsilons: list[float],
    max_iter: int,
    tol: float,
    method: str,
) -> None:
    print("\nBenchmark over fresh random point sets")
    print(f"trials per size = {trials}")
    print(f"sinkhorn method = {method}")
    print(
        "n  exact_ms  exact_cost  eps      sink_ms  sink_cost  "
        "abs_err   rel_err%  assign_overlap  assign_err  iters"
    )
    print("-" * 110)

    for n in sizes:
        exact_times: list[float] = []
        exact_costs: list[float] = []
        sums = {
            epsilon: {
                "time": 0.0,
                "cost": 0.0,
                "abs_err": 0.0,
                "rel_err": 0.0,
                "overlap": 0.0,
                "assign_error": 0.0,
                "iters": 0.0,
            }
            for epsilon in epsilons
        }

        for _ in range(trials):
            x, y = random_points(rng, n)
            cost = cost_matrix(x, y)
            exact, exact_ms = timed_call(exact_assignment, cost)
            exact_times.append(exact_ms)
            exact_costs.append(exact.cost)

            sinkhorn_results = sinkhorn_for_epsilons(
                cost,
                epsilons,
                method,
                max_iter,
                tol,
            )
            for epsilon in epsilons:
                timed_result = sinkhorn_results[epsilon]
                result = timed_result.result
                sink_ms = timed_result.elapsed_ms
                abs_err = result.cost - exact.cost
                rel_err = abs_err / max(abs(exact.cost), 1e-12)
                assignment_overlap, assignment_error = assignment_overlap_and_error(
                    exact,
                    result.plan,
                )
                sums[epsilon]["time"] += sink_ms
                sums[epsilon]["cost"] += result.cost
                sums[epsilon]["abs_err"] += abs_err
                sums[epsilon]["rel_err"] += rel_err
                sums[epsilon]["overlap"] += assignment_overlap
                sums[epsilon]["assign_error"] += assignment_error
                sums[epsilon]["iters"] += result.iterations

        exact_mean = sum(exact_times) / trials
        exact_cost_mean = sum(exact_costs) / trials
        for index, epsilon in enumerate(epsilons):
            prefix = (
                f"{n:<3} {exact_mean:8.3f}  {exact_cost_mean:10.6f}"
                if index == 0
                else " " * 25
            )
            stats = sums[epsilon]
            print(
                f"{prefix}  {epsilon:<7g} "
                f"{stats['time'] / trials:7.3f}  "
                f"{stats['cost'] / trials:9.6f}  "
                f"{stats['abs_err'] / trials:8.6f}  "
                f"{100.0 * stats['rel_err'] / trials:8.1f}  "
                f"{stats['overlap'] / trials:14.6f}  "
                f"{stats['assign_error'] / trials:10.6f}  "
                f"{stats['iters'] / trials:5.1f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare exact tiny assignment with entropy-regularized "
            "Sinkhorn transport on random 1D values."
        )
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--detail-size", type=int, default=5)
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 25, 50, 100, 200])
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[0.5, 0.1, 0.02],
        help="Sinkhorn entropy scales. Smaller means sharper.",
    )
    parser.add_argument("--max-iter", type=int, default=2_000)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument(
        "--method",
        choices=["primal", "log", "scaling"],
        default="primal",
        help=(
            "Use log for very small epsilons. Use scaling to warm-start log "
            "Sinkhorn from larger epsilons to smaller epsilons."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rng = np.random.default_rng(args.seed)
    print_detail(
        rng,
        args.detail_size,
        args.epsilons,
        args.max_iter,
        args.tol,
        args.method,
    )
    print_benchmark(
        rng,
        args.sizes,
        args.trials,
        args.epsilons,
        args.max_iter,
        args.tol,
        args.method,
    )


if __name__ == "__main__":
    main()
