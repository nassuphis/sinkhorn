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

## Rebuild Report

```bash
Rscript --vanilla -e "knitr::knit('sinkhorn.Rnw', output='sinkhorn.tex')"
pdflatex -interaction=nonstopmode sinkhorn.tex
```
