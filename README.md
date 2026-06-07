# Sinkhorn vs Hungarian Assignment

Small benchmark and report comparing Hungarian assignment with Sinkhorn
transport for root-tracking-style matching problems.

## Contents

- `sinkhorn.Rnw`: report source.
- `sinkhorn.pdf`: rendered report.
- `sinkhorn_benchmark.py`: NumPy-only Hungarian/Sinkhorn benchmark.
- `root_tracking_benchmark.py`: synthetic complex-root tracking benchmark.
- `sinkhorn_mt.c`: pthread-based C Sinkhorn and serial C Hungarian routines.
- `sinkhorn_mt_benchmark.py`: Python `ctypes` benchmark wrapper for the C code.
- `image_grid_sinkhorn.py`: image feature extraction, t-SNE embedding, Sinkhorn
  grid snapping, and mosaic rendering.
- `sinkhorn.md`: implementation plan and notes from the sketch phase.

## Python Setup

```bash
uv sync
```

## Run Benchmarks

Python/NumPy benchmark:

```bash
uv run python sinkhorn_benchmark.py
```

C benchmark:

```bash
uv run python sinkhorn_mt_benchmark.py --build --cost-kind sorted-1d
uv run python sinkhorn_mt_benchmark.py --build --cost-kind dense
uv run python sinkhorn_mt_benchmark.py --build --cost-kind flat
```

## Build Image Mosaic

The example Andros thumbnail set is included in `andros_thumbnails/`. To
recreate the mosaics and comparison outputs, run:

```bash
uv run python image_grid_sinkhorn.py \
  --input andros_thumbnails \
  --out outputs/andros_sinkhorn_mosaic.png \
  --layout outputs/andros_sinkhorn_layout.csv \
  --debug outputs/andros_sinkhorn_debug.png \
  --hungarian-debug outputs/andros_hungarian_debug.png
```

The script extracts simple visual features, runs PCA followed by t-SNE, solves a
log-domain Sinkhorn transport problem from embedding coordinates to grid
coordinates, and rounds the soft plan into a one-image-per-cell layout. It also
computes the exact Hungarian grid assignment on the same embedding and writes:

- `outputs/andros_sinkhorn_mosaic.png`
- `outputs/andros_hungarian_mosaic.png`
- `outputs/andros_sinkhorn_debug.png`
- `outputs/andros_hungarian_debug.png`
- `outputs/andros_sinkhorn_vs_hungarian.png`
- `outputs/andros_sinkhorn_vs_hungarian_marked.png`
- `outputs/andros_sinkhorn_vs_hungarian.csv`

In the marked comparison image, red boxes indicate grid cells whose contents
differ between the rounded Sinkhorn layout and the exact Hungarian layout.
The `outputs/` directory is generated and is intentionally not tracked.

### Example Debug Plots

Greedy Sinkhorn rounded assignment:

![Sinkhorn debug assignment](docs/images/andros_sinkhorn_debug.png)

Recursive Sinkhorn repair assignment:

![Recursive Sinkhorn debug assignment](docs/images/andros_recursive_sinkhorn_debug.png)

Hungarian exact assignment:

![Hungarian debug assignment](docs/images/andros_hungarian_debug.png)

### Sinkhorn vs Hungarian Grid Assignment

The image below shows the rounded Sinkhorn layout on the left and the exact
Hungarian layout on the right. Red boxes mark cells where the image differs
between the two assignments.

![Sinkhorn vs Hungarian marked comparison](docs/images/andros_sinkhorn_vs_hungarian_marked.jpg)

## Rebuild Report

```bash
Rscript --vanilla -e "knitr::knit('sinkhorn.Rnw', output='sinkhorn.tex')"
pdflatex -interaction=nonstopmode sinkhorn.tex
```
