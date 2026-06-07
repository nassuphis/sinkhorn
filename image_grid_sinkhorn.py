from __future__ import annotations

import argparse
import contextlib
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

if not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class LoadedImage:
    path: Path
    image: Image.Image
    mean_rgb: tuple[int, int, int]


def parse_size(raw: str) -> tuple[int, int]:
    parts = raw.lower().replace(",", "x").split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("size must look like WIDTHxHEIGHT")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("size values must be positive")
    return width, height


def image_paths(input_dir: Path) -> list[Path]:
    paths = [
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(paths, key=lambda path: path.name.lower())


def load_image(path: Path) -> LoadedImage:
    with Image.open(path) as raw:
        img = ImageOps.exif_transpose(raw).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    mean = tuple(int(round(x)) for x in arr.mean(axis=(0, 1)))
    return LoadedImage(path=path, image=img, mean_rgb=mean)


def rgb_histogram(arr: np.ndarray, bins: int) -> np.ndarray:
    pieces = []
    for channel in range(3):
        hist, _ = np.histogram(arr[:, :, channel], bins=bins, range=(0.0, 1.0))
        hist = hist.astype(np.float32)
        hist /= max(float(hist.sum()), 1.0)
        pieces.append(hist)
    return np.concatenate(pieces)


def spatial_color_means(arr: np.ndarray, grid: int) -> np.ndarray:
    height, width, _ = arr.shape
    pieces = []
    for gy in range(grid):
        y0 = round(gy * height / grid)
        y1 = round((gy + 1) * height / grid)
        for gx in range(grid):
            x0 = round(gx * width / grid)
            x1 = round((gx + 1) * width / grid)
            block = arr[y0:y1, x0:x1, :]
            pieces.append(block.mean(axis=(0, 1)))
    return np.concatenate(pieces).astype(np.float32)


def edge_features(arr: np.ndarray, grid: int) -> np.ndarray:
    gray = (
        0.299 * arr[:, :, 0] +
        0.587 * arr[:, :, 1] +
        0.114 * arr[:, :, 2]
    )
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = np.diff(gray, axis=1)
    gy[1:, :] = np.diff(gray, axis=0)
    mag = np.sqrt(gx * gx + gy * gy)

    height, width = mag.shape
    pieces: list[float] = [
        float(mag.mean()),
        float(mag.std()),
        float(np.quantile(mag, 0.9)),
    ]
    for by in range(grid):
        y0 = round(by * height / grid)
        y1 = round((by + 1) * height / grid)
        for bx in range(grid):
            x0 = round(bx * width / grid)
            x1 = round((bx + 1) * width / grid)
            pieces.append(float(mag[y0:y1, x0:x1].mean()))
    return np.asarray(pieces, dtype=np.float32)


def extract_features(
    loaded: list[LoadedImage],
    lowres_size: tuple[int, int],
    hist_bins: int,
    spatial_grid: int,
) -> np.ndarray:
    features = []
    for item in loaded:
        low = item.image.resize(lowres_size, Image.Resampling.LANCZOS)
        arr = np.asarray(low, dtype=np.float32) / 255.0
        pixels = arr.reshape(-1)
        hist = rgb_histogram(arr, hist_bins)
        spatial = spatial_color_means(arr, spatial_grid)
        edges = edge_features(arr, spatial_grid)
        features.append(np.concatenate([pixels, hist, spatial, edges]))
    return np.vstack(features).astype(np.float32)


def run_embedding(
    features: np.ndarray,
    pca_dims: int,
    perplexity: float,
    tsne_iter: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n, feature_dim = features.shape
    scaled = StandardScaler().fit_transform(features)
    dims = min(pca_dims, feature_dim, n - 1)
    if dims < 2:
        raise ValueError("need at least three images for PCA/t-SNE")

    reduced = PCA(n_components=dims, random_state=seed).fit_transform(scaled)
    safe_perplexity = min(perplexity, max(5.0, (n - 1) / 3.0))
    safe_perplexity = min(safe_perplexity, n - 1.0)

    kwargs = dict(
        n_components=2,
        perplexity=safe_perplexity,
        learning_rate="auto",
        init="pca",
        metric="euclidean",
        random_state=seed,
        n_jobs=1,
    )
    with open(os.devnull, "w") as stderr, contextlib.redirect_stderr(stderr):
        try:
            embedding = TSNE(max_iter=tsne_iter, **kwargs).fit_transform(reduced)
        except TypeError:
            kwargs.pop("n_jobs", None)
            embedding = TSNE(n_iter=tsne_iter, **kwargs).fit_transform(reduced)

    return reduced, embedding.astype(np.float64)


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    lo = embedding.min(axis=0)
    hi = embedding.max(axis=0)
    span = np.maximum(hi - lo, 1e-12)
    return (embedding - lo) / span


def infer_grid(n: int, image_aspect: float) -> tuple[int, int]:
    cols = max(1, int(math.ceil(math.sqrt(n * image_aspect))))
    rows = int(math.ceil(n / cols))
    while rows * cols < n:
        cols += 1
        rows = int(math.ceil(n / cols))
    return rows, cols


def grid_coordinates(rows: int, cols: int) -> np.ndarray:
    coords = []
    for row in range(rows):
        y = 0.5 if rows == 1 else row / (rows - 1)
        for col in range(cols):
            x = 0.5 if cols == 1 else col / (cols - 1)
            coords.append((x, y))
    return np.asarray(coords, dtype=np.float64)


def auto_epsilon(cost: np.ndarray, scale: float) -> float:
    sorted_cost = np.sort(cost, axis=1)
    if sorted_cost.shape[1] >= 2:
        gaps = sorted_cost[:, 1] - sorted_cost[:, 0]
        base = float(np.median(gaps[gaps > 0])) if np.any(gaps > 0) else 0.0
    else:
        base = 0.0
    if base <= 0:
        base = float(np.median(cost))
    if base <= 0:
        base = 1e-3
    return max(scale * base, 1e-8)


def logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    shifted = np.exp(values - max_values)
    sums = np.sum(shifted, axis=axis, keepdims=True)
    return np.squeeze(max_values + np.log(sums), axis=axis)


def sinkhorn_log(
    cost: np.ndarray,
    epsilon: float,
    iterations: int,
    tol: float,
) -> tuple[np.ndarray, float, float, int]:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    n, m = cost.shape
    log_a = np.full(n, -math.log(n), dtype=np.float64)
    log_b = np.full(m, -math.log(m), dtype=np.float64)
    log_k = -cost / epsilon
    log_u = np.zeros(n, dtype=np.float64)
    log_v = np.zeros(m, dtype=np.float64)

    row_error = math.inf
    col_error = math.inf
    plan = np.empty_like(cost)
    for step in range(1, iterations + 1):
        log_u = log_a - logsumexp(log_k + log_v[None, :], axis=1)
        log_v = log_b - logsumexp(log_k + log_u[:, None], axis=0)

        if step % 25 == 0 or step == iterations:
            log_p = log_u[:, None] + log_k + log_v[None, :]
            plan = np.exp(log_p)
            row_error = float(np.max(np.abs(plan.sum(axis=1) - np.exp(log_a))))
            col_error = float(np.max(np.abs(plan.sum(axis=0) - np.exp(log_b))))
            if max(row_error, col_error) <= tol:
                return plan, row_error, col_error, step

    log_p = log_u[:, None] + log_k + log_v[None, :]
    plan = np.exp(log_p)
    row_error = float(np.max(np.abs(plan.sum(axis=1) - np.exp(log_a))))
    col_error = float(np.max(np.abs(plan.sum(axis=0) - np.exp(log_b))))
    return plan, row_error, col_error, iterations


def augment_cost(cost: np.ndarray, cell_count: int) -> np.ndarray:
    image_count = cost.shape[0]
    if image_count > cell_count:
        raise ValueError("grid has fewer cells than images")
    if image_count == cell_count:
        return cost
    dummy_count = cell_count - image_count
    dummy_cost = float(np.median(cost))
    dummy = np.full((dummy_count, cell_count), dummy_cost, dtype=np.float64)
    return np.vstack([cost, dummy])


def greedy_round(
    plan: np.ndarray,
    cost: np.ndarray,
    image_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    scores = plan[:image_count, :].copy()
    jitter = rng.random(scores.shape) * 1e-12
    scores += jitter

    sorted_scores = np.sort(scores, axis=1)
    top = sorted_scores[:, -1]
    second = sorted_scores[:, -2] if scores.shape[1] >= 2 else np.zeros(image_count)
    margin = top - second
    order = np.lexsort((-top, -margin))

    assigned = np.full(image_count, -1, dtype=int)
    free = np.ones(scores.shape[1], dtype=bool)
    for image_idx in order:
        preferences = np.argsort(scores[image_idx])[::-1]
        chosen = -1
        for cell_idx in preferences:
            if free[cell_idx]:
                chosen = int(cell_idx)
                break
        if chosen < 0:
            free_indices = np.flatnonzero(free)
            chosen = int(free_indices[np.argmin(cost[image_idx, free_indices])])
        assigned[image_idx] = chosen
        free[chosen] = False

    assigned_score = plan[np.arange(image_count), assigned]
    assigned_cost = cost[np.arange(image_count), assigned]
    return assigned, assigned_score, assigned_cost, margin


def recursive_sinkhorn_round(
    plan: np.ndarray,
    cost: np.ndarray,
    image_count: int,
    seed: int,
    epsilon: float,
    iterations: int,
    tol: float,
    max_rounds: int,
    candidates_per_image: int,
    repair_epsilon_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    rng = np.random.default_rng(seed)
    scores = plan[:image_count, :].copy()
    scores += rng.random(scores.shape) * 1e-12

    sorted_scores = np.sort(scores, axis=1)
    top = sorted_scores[:, -1]
    second = sorted_scores[:, -2] if scores.shape[1] >= 2 else np.zeros(image_count)
    margin = top - second

    assigned = np.argmax(scores, axis=1).astype(int)
    cell_count = scores.shape[1]
    counts = np.bincount(assigned, minlength=cell_count)
    stats = {
        "initial_duplicate_cells": int(np.sum(counts > 1)),
        "initial_conflicted_images": int(np.sum(counts[assigned] > 1)),
        "repair_rounds": 0,
        "local_repairs": 0,
        "argmax_duplicate_rounds": 0,
        "final_hungarian_fallbacks": 0,
        "max_local_images": 0,
        "max_local_cells": 0,
    }
    repair_scale = repair_epsilon_scale if repair_epsilon_scale > 0 else 1.0

    for repair_round in range(1, max_rounds + 1):
        counts = np.bincount(assigned, minlength=cell_count)
        duplicate_cells = np.flatnonzero(counts > 1)
        if duplicate_cells.size == 0:
            break

        repair_images = np.flatnonzero(np.isin(assigned, duplicate_cells))
        stable = np.ones(image_count, dtype=bool)
        stable[repair_images] = False

        locked_cells = np.zeros(cell_count, dtype=bool)
        locked_cells[assigned[stable]] = True
        available_cells = np.flatnonzero(~locked_cells)
        available_mask = np.zeros(cell_count, dtype=bool)
        available_mask[available_cells] = True

        candidate_mask = np.zeros(cell_count, dtype=bool)
        candidate_mask[duplicate_cells] = True

        if candidates_per_image > 0 and available_cells.size > 0:
            top_count = min(candidates_per_image, available_cells.size)
            available_scores = scores[np.ix_(repair_images, available_cells)]
            for row_scores in available_scores:
                top_local = np.argsort(row_scores)[-top_count:]
                candidate_mask[available_cells[top_local]] = True

        candidate_mask &= available_mask
        if int(np.sum(candidate_mask)) < repair_images.size:
            aggregate = np.max(scores[np.ix_(repair_images, available_cells)], axis=0)
            for local_idx in np.argsort(aggregate)[::-1]:
                candidate_mask[available_cells[local_idx]] = True
                if int(np.sum(candidate_mask)) >= repair_images.size:
                    break

        candidate_cells = np.flatnonzero(candidate_mask)
        if candidate_cells.size < repair_images.size:
            candidate_cells = available_cells

        local_cost = cost[np.ix_(repair_images, candidate_cells)]
        repair_epsilon = max(epsilon * (repair_scale ** repair_round), 1e-8)
        local_plan, _, _, _ = sinkhorn_log(
            augment_cost(local_cost, candidate_cells.size),
            epsilon=repair_epsilon,
            iterations=iterations,
            tol=tol,
        )
        local_scores = local_plan[:repair_images.size, :].copy()
        local_scores += rng.random(local_scores.shape) * 1e-12
        proposed = candidate_cells[np.argmax(local_scores, axis=1)]

        if np.unique(proposed).size < proposed.size:
            stats["argmax_duplicate_rounds"] += 1

        assigned[repair_images] = proposed
        stats["repair_rounds"] = repair_round
        stats["local_repairs"] += 1
        stats["max_local_images"] = max(stats["max_local_images"], int(repair_images.size))
        stats["max_local_cells"] = max(stats["max_local_cells"], int(candidate_cells.size))

    counts = np.bincount(assigned, minlength=cell_count)
    if np.any(counts > 1):
        duplicate_cells = np.flatnonzero(counts > 1)
        repair_images = np.flatnonzero(np.isin(assigned, duplicate_cells))
        stable = np.ones(image_count, dtype=bool)
        stable[repair_images] = False
        locked_cells = np.zeros(cell_count, dtype=bool)
        locked_cells[assigned[stable]] = True
        candidate_cells = np.flatnonzero(~locked_cells)
        local_cost = cost[np.ix_(repair_images, candidate_cells)]
        local_assigned, _ = hungarian_assign(local_cost)
        assigned[repair_images] = candidate_cells[local_assigned]
        stats["final_hungarian_fallbacks"] += 1
        stats["local_repairs"] += 1
        stats["max_local_images"] = max(stats["max_local_images"], int(repair_images.size))
        stats["max_local_cells"] = max(stats["max_local_cells"], int(candidate_cells.size))

    assigned_score = plan[np.arange(image_count), assigned]
    assigned_cost = cost[np.arange(image_count), assigned]
    return assigned, assigned_score, assigned_cost, margin, stats


def hungarian_assign(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    row_ind, col_ind = linear_sum_assignment(cost)
    assigned = np.full(cost.shape[0], -1, dtype=int)
    assigned[row_ind] = col_ind
    if np.any(assigned < 0):
        raise RuntimeError("Hungarian assignment did not assign every image")
    assigned_cost = cost[np.arange(cost.shape[0]), assigned]
    return assigned, assigned_cost


def render_mosaic(
    loaded: list[LoadedImage],
    assigned: np.ndarray,
    rows: int,
    cols: int,
    cell_size: tuple[int, int],
    gap: int,
    out_path: Path,
    highlight_cells: np.ndarray | None = None,
) -> None:
    cell_w, cell_h = cell_size
    width = cols * cell_w + (cols + 1) * gap
    height = rows * cell_h + (rows + 1) * gap
    canvas = Image.new("RGB", (width, height), (245, 245, 242))
    draw = ImageDraw.Draw(canvas)

    for row in range(rows):
        for col in range(cols):
            x0 = gap + col * (cell_w + gap)
            y0 = gap + row * (cell_h + gap)
            draw.rectangle(
                [x0, y0, x0 + cell_w - 1, y0 + cell_h - 1],
                fill=(232, 232, 228),
            )

    for image_idx, cell_idx in enumerate(assigned):
        row = int(cell_idx // cols)
        col = int(cell_idx % cols)
        x0 = gap + col * (cell_w + gap)
        y0 = gap + row * (cell_h + gap)
        thumb = ImageOps.fit(
            loaded[image_idx].image,
            (cell_w, cell_h),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        canvas.paste(thumb, (x0, y0))

    if highlight_cells is not None:
        for cell_idx, highlight in enumerate(highlight_cells):
            if not highlight:
                continue
            row = int(cell_idx // cols)
            col = int(cell_idx % cols)
            x0 = gap + col * (cell_w + gap)
            y0 = gap + row * (cell_h + gap)
            for inset in range(3):
                draw.rectangle(
                    [
                        x0 + inset,
                        y0 + inset,
                        x0 + cell_w - 1 - inset,
                        y0 + cell_h - 1 - inset,
                    ],
                    outline=(230, 40, 35),
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def render_side_by_side(
    left_path: Path,
    right_path: Path,
    out_path: Path,
    left_label: str = "Sinkhorn rounded",
    right_label: str = "Hungarian exact",
) -> None:
    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")
    label_h = 46
    gap = 16
    width = left.width + right.width + gap
    height = max(left.height, right.height) + label_h
    canvas = Image.new("RGB", (width, height), (250, 250, 248))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 14), left_label, fill=(20, 20, 20))
    draw.text((left.width + gap + 12, 14), right_label, fill=(20, 20, 20))
    canvas.paste(left, (0, label_h))
    canvas.paste(right, (left.width + gap, label_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def changed_cells(
    sinkhorn_assigned: np.ndarray,
    hungarian_assigned: np.ndarray,
    cell_count: int,
) -> np.ndarray:
    sink_by_cell = np.full(cell_count, -1, dtype=int)
    hung_by_cell = np.full(cell_count, -1, dtype=int)
    for image_idx, cell_idx in enumerate(sinkhorn_assigned):
        sink_by_cell[cell_idx] = image_idx
    for image_idx, cell_idx in enumerate(hungarian_assigned):
        hung_by_cell[cell_idx] = image_idx
    return sink_by_cell != hung_by_cell


def render_debug(
    embedding: np.ndarray,
    grid: np.ndarray,
    assigned: np.ndarray,
    mean_colors: list[tuple[int, int, int]],
    out_path: Path,
) -> None:
    width, height = 1200, 900
    pad = 55
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    def point(coord: np.ndarray) -> tuple[int, int]:
        x = int(round(pad + coord[0] * (width - 2 * pad)))
        y = int(round(pad + coord[1] * (height - 2 * pad)))
        return x, y

    for coord in grid:
        x, y = point(coord)
        draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(215, 215, 215))

    for image_idx, cell_idx in enumerate(assigned):
        color = mean_colors[image_idx]
        ex, ey = point(embedding[image_idx])
        gx, gy = point(grid[cell_idx])
        line_color = tuple(int(0.70 * c + 0.30 * 255) for c in color)
        draw.line([ex, ey, gx, gy], fill=line_color, width=1)

    for image_idx, coord in enumerate(embedding):
        x, y = point(coord)
        color = mean_colors[image_idx]
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=color, outline=(20, 20, 20))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def write_layout(
    out_path: Path,
    loaded: list[LoadedImage],
    embedding_raw: np.ndarray,
    embedding: np.ndarray,
    grid: np.ndarray,
    assigned: np.ndarray,
    assigned_score: np.ndarray,
    assigned_cost: np.ndarray,
    top_margin: np.ndarray,
    rows: int,
    cols: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "image_index",
            "filename",
            "path",
            "tsne_x",
            "tsne_y",
            "embed_x",
            "embed_y",
            "grid_row",
            "grid_col",
            "grid_x",
            "grid_y",
            "sinkhorn_assigned_score",
            "sinkhorn_margin",
            "assignment_cost",
            "mean_r",
            "mean_g",
            "mean_b",
        ])
        for image_idx, item in enumerate(loaded):
            cell_idx = int(assigned[image_idx])
            row = cell_idx // cols
            col = cell_idx % cols
            mean_r, mean_g, mean_b = item.mean_rgb
            writer.writerow([
                image_idx,
                item.path.name,
                str(item.path),
                f"{embedding_raw[image_idx, 0]:.8f}",
                f"{embedding_raw[image_idx, 1]:.8f}",
                f"{embedding[image_idx, 0]:.8f}",
                f"{embedding[image_idx, 1]:.8f}",
                row,
                col,
                f"{grid[cell_idx, 0]:.8f}",
                f"{grid[cell_idx, 1]:.8f}",
                f"{assigned_score[image_idx]:.12g}",
                f"{top_margin[image_idx]:.12g}",
                f"{assigned_cost[image_idx]:.12g}",
                mean_r,
                mean_g,
                mean_b,
            ])


def write_comparison(
    out_path: Path,
    loaded: list[LoadedImage],
    grid: np.ndarray,
    sinkhorn_assigned: np.ndarray,
    sinkhorn_score: np.ndarray,
    sinkhorn_cost: np.ndarray,
    sinkhorn_margin: np.ndarray,
    hungarian_assigned: np.ndarray,
    hungarian_cost: np.ndarray,
    rows: int,
    cols: int,
) -> dict[str, float]:
    same_cell = sinkhorn_assigned == hungarian_assigned
    grid_delta = grid[sinkhorn_assigned] - grid[hungarian_assigned]
    euclid = np.sqrt(np.sum(grid_delta * grid_delta, axis=1))
    manhattan_cells = (
        np.abs(sinkhorn_assigned // cols - hungarian_assigned // cols) +
        np.abs(sinkhorn_assigned % cols - hungarian_assigned % cols)
    )
    cost_delta = sinkhorn_cost - hungarian_cost
    stats = {
        "same_cells": float(np.sum(same_cell)),
        "same_cell_fraction": float(np.mean(same_cell)),
        "mean_sinkhorn_cost": float(np.mean(sinkhorn_cost)),
        "mean_hungarian_cost": float(np.mean(hungarian_cost)),
        "total_sinkhorn_cost": float(np.sum(sinkhorn_cost)),
        "total_hungarian_cost": float(np.sum(hungarian_cost)),
        "mean_cost_delta": float(np.mean(cost_delta)),
        "mean_grid_euclidean_move": float(np.mean(euclid)),
        "mean_grid_manhattan_cells": float(np.mean(manhattan_cells)),
        "max_grid_manhattan_cells": float(np.max(manhattan_cells)),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "image_index",
            "filename",
            "sinkhorn_row",
            "sinkhorn_col",
            "hungarian_row",
            "hungarian_col",
            "same_cell",
            "grid_euclidean_move",
            "grid_manhattan_cells",
            "sinkhorn_score",
            "sinkhorn_margin",
            "sinkhorn_cost",
            "hungarian_cost",
            "cost_delta",
        ])
        for image_idx, item in enumerate(loaded):
            s_cell = int(sinkhorn_assigned[image_idx])
            h_cell = int(hungarian_assigned[image_idx])
            writer.writerow([
                image_idx,
                item.path.name,
                s_cell // cols,
                s_cell % cols,
                h_cell // cols,
                h_cell % cols,
                int(same_cell[image_idx]),
                f"{euclid[image_idx]:.12g}",
                int(manhattan_cells[image_idx]),
                f"{sinkhorn_score[image_idx]:.12g}",
                f"{sinkhorn_margin[image_idx]:.12g}",
                f"{sinkhorn_cost[image_idx]:.12g}",
                f"{hungarian_cost[image_idx]:.12g}",
                f"{cost_delta[image_idx]:.12g}",
            ])
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed thumbnails with PCA/t-SNE and snap them to a grid with Sinkhorn."
    )
    parser.add_argument("--input", type=Path, default=Path("andros_thumbnails"))
    parser.add_argument("--out", type=Path, default=Path("outputs/andros_sinkhorn_mosaic.png"))
    parser.add_argument("--layout", type=Path, default=Path("outputs/andros_sinkhorn_layout.csv"))
    parser.add_argument("--debug", type=Path, default=Path("outputs/andros_sinkhorn_debug.png"))
    parser.add_argument("--hungarian-out", type=Path, default=Path("outputs/andros_hungarian_mosaic.png"))
    parser.add_argument("--hungarian-layout", type=Path, default=Path("outputs/andros_hungarian_layout.csv"))
    parser.add_argument("--hungarian-debug", type=Path, default=Path("outputs/andros_hungarian_debug.png"))
    parser.add_argument("--compare-out", type=Path, default=Path("outputs/andros_sinkhorn_vs_hungarian.png"))
    parser.add_argument("--marked-compare-out", type=Path, default=Path("outputs/andros_sinkhorn_vs_hungarian_marked.png"))
    parser.add_argument("--compare-layout", type=Path, default=Path("outputs/andros_sinkhorn_vs_hungarian.csv"))
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--cols", type=int, default=None)
    parser.add_argument("--cell-size", type=parse_size, default=(120, 90))
    parser.add_argument("--gap", type=int, default=2)
    parser.add_argument("--lowres-size", type=parse_size, default=(24, 18))
    parser.add_argument("--hist-bins", type=int, default=8)
    parser.add_argument("--spatial-grid", type=int, default=4)
    parser.add_argument("--pca-dims", type=int, default=40)
    parser.add_argument("--perplexity", type=float, default=35.0)
    parser.add_argument("--tsne-iter", type=int, default=1500)
    parser.add_argument("--sinkhorn-iterations", type=int, default=1500)
    parser.add_argument("--sinkhorn-tol", type=float, default=1e-8)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--epsilon-scale", type=float, default=1.0)
    parser.add_argument(
        "--rounding",
        choices=("greedy", "recursive-sinkhorn"),
        default="greedy",
        help="How to turn the soft Sinkhorn plan into a one-image-per-cell layout.",
    )
    parser.add_argument(
        "--repair-rounds",
        type=int,
        default=6,
        help="Maximum duplicate-repair rounds for recursive Sinkhorn rounding.",
    )
    parser.add_argument(
        "--repair-candidates",
        type=int,
        default=12,
        help="Top available cells per conflicted image in each repair subproblem.",
    )
    parser.add_argument(
        "--repair-epsilon-scale",
        type=float,
        default=0.5,
        help="Multiplier applied to epsilon for local recursive Sinkhorn repairs.",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    paths = image_paths(args.input)
    if not paths:
        raise SystemExit(f"no images found in {args.input}")

    print(f"loading {len(paths)} images from {args.input}")
    loaded = [load_image(path) for path in paths]

    cell_aspect = args.cell_size[0] / args.cell_size[1]
    rows = args.rows
    cols = args.cols
    if rows is None and cols is None:
        rows, cols = infer_grid(len(loaded), cell_aspect)
    elif rows is None:
        rows = int(math.ceil(len(loaded) / cols))
    elif cols is None:
        cols = int(math.ceil(len(loaded) / rows))
    if rows <= 0 or cols <= 0:
        raise SystemExit("rows and cols must be positive")
    if rows * cols < len(loaded):
        raise SystemExit(f"grid {rows}x{cols} has fewer cells than images")

    print(f"extracting features")
    features = extract_features(
        loaded,
        lowres_size=args.lowres_size,
        hist_bins=args.hist_bins,
        spatial_grid=args.spatial_grid,
    )
    print(f"feature matrix = {features.shape[0]} x {features.shape[1]}")

    print("running PCA/t-SNE")
    _, embedding_raw = run_embedding(
        features,
        pca_dims=args.pca_dims,
        perplexity=args.perplexity,
        tsne_iter=args.tsne_iter,
        seed=args.seed,
    )
    embedding = normalize_embedding(embedding_raw)
    grid = grid_coordinates(rows, cols)

    cost = np.sum((embedding[:, None, :] - grid[None, :, :]) ** 2, axis=2)
    epsilon = args.epsilon
    if epsilon is None:
        epsilon = auto_epsilon(cost, args.epsilon_scale)
    print(f"grid = {rows} x {cols} ({rows * cols} cells), epsilon = {epsilon:.6g}")

    print("running log-domain Sinkhorn")
    augmented_cost = augment_cost(cost, rows * cols)
    plan, row_error, col_error, used_iterations = sinkhorn_log(
        augmented_cost,
        epsilon=epsilon,
        iterations=args.sinkhorn_iterations,
        tol=args.sinkhorn_tol,
    )
    print(
        "sinkhorn iterations = "
        f"{used_iterations}, row_err = {row_error:.2e}, col_err = {col_error:.2e}"
    )

    repair_stats: dict[str, int] | None = None
    if args.rounding == "greedy":
        assigned, assigned_score, assigned_cost, top_margin = greedy_round(
            plan,
            cost,
            image_count=len(loaded),
            seed=args.seed,
        )
    else:
        assigned, assigned_score, assigned_cost, top_margin, repair_stats = (
            recursive_sinkhorn_round(
                plan,
                cost,
                image_count=len(loaded),
                seed=args.seed,
                epsilon=epsilon,
                iterations=args.sinkhorn_iterations,
                tol=args.sinkhorn_tol,
                max_rounds=args.repair_rounds,
                candidates_per_image=args.repair_candidates,
                repair_epsilon_scale=args.repair_epsilon_scale,
            )
        )
    free_cells = rows * cols - len(np.unique(assigned))
    print(
        f"{args.rounding} assignment: {len(loaded)} images, "
        f"{free_cells} blank cells, mean cost = {assigned_cost.mean():.6g}"
    )
    if repair_stats is not None:
        print(
            "recursive repair: "
            f"initial duplicate cells = {repair_stats['initial_duplicate_cells']}, "
            f"conflicted images = {repair_stats['initial_conflicted_images']}, "
            f"repair rounds = {repair_stats['repair_rounds']}, "
            f"max local problem = {repair_stats['max_local_images']} x "
            f"{repair_stats['max_local_cells']}, "
            f"duplicate argmax rounds = {repair_stats['argmax_duplicate_rounds']}, "
            f"final Hungarian fallbacks = {repair_stats['final_hungarian_fallbacks']}"
        )
    print(
        f"confidence: mean assigned score = {assigned_score.mean():.6g}, "
        f"mean margin = {top_margin.mean():.6g}"
    )

    print("running exact Hungarian assignment")
    hungarian_assigned, hungarian_cost = hungarian_assign(cost)
    print(
        f"hungarian: mean cost = {hungarian_cost.mean():.6g}, "
        f"total cost = {hungarian_cost.sum():.6g}"
    )

    render_mosaic(
        loaded,
        assigned,
        rows=rows,
        cols=cols,
        cell_size=args.cell_size,
        gap=args.gap,
        out_path=args.out,
    )
    render_mosaic(
        loaded,
        hungarian_assigned,
        rows=rows,
        cols=cols,
        cell_size=args.cell_size,
        gap=args.gap,
        out_path=args.hungarian_out,
    )
    render_debug(
        embedding,
        grid,
        assigned,
        [item.mean_rgb for item in loaded],
        args.debug,
    )
    render_debug(
        embedding,
        grid,
        hungarian_assigned,
        [item.mean_rgb for item in loaded],
        args.hungarian_debug,
    )
    write_layout(
        args.layout,
        loaded,
        embedding_raw,
        embedding,
        grid,
        assigned,
        assigned_score,
        assigned_cost,
        top_margin,
        rows=rows,
        cols=cols,
    )
    write_layout(
        args.hungarian_layout,
        loaded,
        embedding_raw,
        embedding,
        grid,
        hungarian_assigned,
        np.full(len(loaded), np.nan),
        hungarian_cost,
        np.full(len(loaded), np.nan),
        rows=rows,
        cols=cols,
    )
    comparison_stats = write_comparison(
        args.compare_layout,
        loaded,
        grid,
        assigned,
        assigned_score,
        assigned_cost,
        top_margin,
        hungarian_assigned,
        hungarian_cost,
        rows=rows,
        cols=cols,
    )
    changed = changed_cells(assigned, hungarian_assigned, rows * cols)
    marked_sinkhorn = args.compare_out.with_name(args.compare_out.stem + "_sinkhorn_marked.png")
    marked_hungarian = args.compare_out.with_name(args.compare_out.stem + "_hungarian_marked.png")
    render_mosaic(
        loaded,
        assigned,
        rows=rows,
        cols=cols,
        cell_size=args.cell_size,
        gap=args.gap,
        out_path=marked_sinkhorn,
        highlight_cells=changed,
    )
    render_mosaic(
        loaded,
        hungarian_assigned,
        rows=rows,
        cols=cols,
        cell_size=args.cell_size,
        gap=args.gap,
        out_path=marked_hungarian,
        highlight_cells=changed,
    )
    sinkhorn_label = (
        "Sinkhorn greedy"
        if args.rounding == "greedy"
        else "Sinkhorn recursive repair"
    )
    render_side_by_side(
        args.out,
        args.hungarian_out,
        args.compare_out,
        left_label=sinkhorn_label,
    )
    render_side_by_side(
        marked_sinkhorn,
        marked_hungarian,
        args.marked_compare_out,
        left_label=f"{sinkhorn_label} (red cells differ)",
        right_label="Hungarian exact (red cells differ)",
    )

    print(f"wrote {args.out}")
    print(f"wrote {args.layout}")
    print(f"wrote {args.debug}")
    print(f"wrote {args.hungarian_out}")
    print(f"wrote {args.hungarian_layout}")
    print(f"wrote {args.hungarian_debug}")
    print(f"wrote {args.compare_out}")
    print(f"wrote {args.marked_compare_out}")
    print(f"wrote {args.compare_layout}")
    print(
        "comparison: "
        f"{comparison_stats['same_cells']:.0f}/{len(loaded)} same cells "
        f"({comparison_stats['same_cell_fraction']:.1%}), "
        f"mean grid move = {comparison_stats['mean_grid_manhattan_cells']:.2f} cells, "
        f"mean cost delta = {comparison_stats['mean_cost_delta']:.6g}"
    )


if __name__ == "__main__":
    main()
