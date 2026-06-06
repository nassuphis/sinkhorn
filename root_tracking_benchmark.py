from __future__ import annotations

import argparse
import time

import numpy as np

import sinkhorn_benchmark as ot


def synthetic_roots(
    rng: np.random.Generator,
    degree: int,
    jitter: float,
    step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angles = 2.0 * np.pi * np.arange(degree) / degree
    old = np.exp(1j * angles)
    old = old + jitter * (rng.normal(size=degree) + 1j * rng.normal(size=degree))

    new_unshuffled = old + step * (
        rng.normal(size=degree) + 1j * rng.normal(size=degree)
    )
    shuffle = rng.permutation(degree)
    new = new_unshuffled[shuffle]

    true_perm = np.empty(degree, dtype=int)
    for new_col, old_i in enumerate(shuffle):
        true_perm[old_i] = new_col

    return old, new, true_perm


def root_cost_matrix(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    return np.abs(old[:, None] - new[None, :]) ** 2


def row_metrics(
    soft_assignment: np.ndarray,
    hungarian_perm: tuple[int, ...],
    true_perm: np.ndarray,
) -> dict[str, float]:
    row_argmax = np.argmax(soft_assignment, axis=1)
    top = np.max(soft_assignment, axis=1)
    second = np.partition(soft_assignment, -2, axis=1)[:, -2]
    hungarian_perm_array = np.array(hungarian_perm)

    return {
        "argmax_hungarian": float(np.mean(row_argmax == hungarian_perm_array)),
        "argmax_true": float(np.mean(row_argmax == true_perm)),
        "unique_cols": float(len(set(int(col) for col in row_argmax))),
        "mean_top": float(np.mean(top)),
        "min_top": float(np.min(top)),
        "mean_margin": float(np.mean(top - second)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic complex-root tracking benchmark."
    )
    parser.add_argument("--degree", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--jitter", type=float, default=0.05)
    parser.add_argument("--step", type=float, default=0.015)
    parser.add_argument(
        "--epsilon-scales",
        type=float,
        nargs="+",
        default=[1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01],
        help="Multipliers of the median nearest-vs-second-nearest cost gap.",
    )
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    old, new, true_perm = synthetic_roots(
        rng,
        args.degree,
        args.jitter,
        args.step,
    )
    cost = root_cost_matrix(old, new)
    exact, exact_ms = ot.timed_call(ot.exact_assignment, cost)
    exact_true_accuracy = float(np.mean(np.array(exact.perm) == true_perm))

    sorted_cost = np.sort(cost, axis=1)
    median_gap = float(np.median(sorted_cost[:, 1] - sorted_cost[:, 0]))

    print(f"degree = {args.degree}")
    print(f"seed = {args.seed}")
    print(f"jitter = {args.jitter}")
    print(f"step = {args.step}")
    print(f"median nearest gap = {median_gap:.8g}")
    print(f"Hungarian time = {exact_ms:.3f} ms")
    print(f"Hungarian vs known synthetic truth = {exact_true_accuracy:.3f}")
    print()
    print(
        "scale  epsilon       sink_ms  argmax_hung  argmax_true  unique_cols  "
        "assign_err  mean_top  min_top  mean_margin  row_err"
    )
    print("-" * 116)

    for scale in args.epsilon_scales:
        epsilon = scale * median_gap
        start = time.perf_counter()
        result = ot.sinkhorn(
            cost,
            epsilon,
            max_iter=args.iterations,
            tol=0.0,
        )
        sink_ms = (time.perf_counter() - start) * 1_000.0
        soft_assignment = result.plan * args.degree
        _, assignment_error = ot.assignment_overlap_and_error(exact, result.plan)
        metrics = row_metrics(soft_assignment, exact.perm, true_perm)

        print(
            f"{scale:<5g}  {epsilon:<12.6g} "
            f"{sink_ms:7.3f}  "
            f"{metrics['argmax_hungarian']:12.3f}  "
            f"{metrics['argmax_true']:11.3f}  "
            f"{metrics['unique_cols']:11.0f}  "
            f"{assignment_error:10.6f}  "
            f"{metrics['mean_top']:8.4f}  "
            f"{metrics['min_top']:7.4f}  "
            f"{metrics['mean_margin']:11.4f}  "
            f"{result.row_error:.2e}"
        )


if __name__ == "__main__":
    main()
