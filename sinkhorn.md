# Plan: NumPy Hungarian vs Sinkhorn Demo

## Goal

Build a tiny, readable Python demo that compares a hard optimal assignment against
an entropic Sinkhorn transport plan for a small set of numbers.

Constraints:

- Use only Python standard library and NumPy.
- No SciPy, POT, JAX, CuPy, GPU code, or external solvers.
- Keep the example small enough that the whole algorithm can be understood from
  one file.
- Prefer clarity over scalability.

## Source Notes

The paper `sinkhorm.pdf` is Cuturi's "Sinkhorn Distances: Lightspeed Computation
of Optimal Transportation Distances".

The relevant implementation idea is Section 4:

- Classical optimal transport solves a linear program over transport matrices.
- Sinkhorn adds entropy regularization, making the optimum smooth and unique.
- The optimal regularized plan has the form:

```text
P = diag(u) K diag(v)
K = exp(-lambda * M)
```

- `M` is the cost matrix.
- `K` is the Gibbs kernel.
- `u` and `v` are scaling vectors found by repeatedly normalizing rows and
  columns so the plan has the desired marginals.

For the demo, use the equivalent `epsilon` form:

```text
K = exp(-M / epsilon)
```

Small `epsilon` gives a sharper plan closer to hard assignment. Large `epsilon`
gives a blurrier, higher-entropy plan.

## Exact Baseline

For tiny examples, the exact assignment can be checked by enumerating all
permutations with `itertools.permutations`.

For benchmarks like `n=200`, permutation enumeration is impossible. The
implemented script therefore uses a dependency-free O(n^3) Hungarian algorithm
for the exact hard assignment baseline, and keeps the brute-force version only as
a small sanity helper.

This still satisfies the simple dependency target: standard library plus NumPy.

## Demo Data

Use two small one-dimensional point sets:

```python
x = np.array([0.0, 1.0, 4.0, 7.0])
y = np.array([0.2, 1.4, 3.8, 8.0])
```

Build the cost matrix with squared distance:

```python
C = (x[:, None] - y[None, :]) ** 2
```

Use uniform marginals:

```python
a = np.ones(n) / n
b = np.ones(n) / n
```

This makes the comparison clean:

- Hungarian baseline: each source point sends all of its mass to one target.
- Sinkhorn: each source point can split mass across targets.

## File Created

The benchmark script is:

```text
sinkhorn_benchmark.py
```

No package structure is needed.

## Functions

### `cost_matrix(x, y)`

Return the pairwise squared-distance matrix:

```python
return (x[:, None] - y[None, :]) ** 2
```

### `exact_assignment(C)`

Use the standard shortest-augmenting-path form of the Hungarian algorithm.

Return:

- `perm`
- assignment matrix `P`, with `P[i, perm[i]] = 1 / n`
- OT-style average cost `np.sum(P * C)`

The divide-by-`n` assignment matrix makes the cost directly comparable to
Sinkhorn's transport cost.

### `sinkhorn(C, epsilon=0.5, max_iter=500, tol=1e-9)`

Implement the basic primal-domain Sinkhorn-Knopp iteration:

```python
K = np.exp(-C / epsilon)
u = np.ones(n)
v = np.ones(n)

for step in range(max_iter):
    u_prev = u.copy()
    u = a / (K @ v)
    v = b / (K.T @ u)

    if np.linalg.norm(u - u_prev, ord=1) < tol:
        break

P = (u[:, None] * K) * v[None, :]
cost = np.sum(P * C)
```

Return:

- transport plan `P`
- cost
- iteration count
- row-sum error `np.max(np.abs(P.sum(axis=1) - a))`
- column-sum error `np.max(np.abs(P.sum(axis=0) - b))`

For this tiny demo, primal-domain Sinkhorn is enough. No log-domain
stabilization is needed unless `epsilon` is made very small.

### `entropy(P)`

Report plan entropy so the soft vs hard behavior is visible:

```python
positive = P[P > 0]
return -np.sum(positive * np.log(positive))
```

### `main()`

Print:

- input point sets
- cost matrix
- exact assignment permutation and plan
- Sinkhorn plans for a few epsilon values, for example:

```python
epsilons = [2.0, 0.5, 0.1]
```

Expected behavior:

- `epsilon = 2.0`: soft, diffuse transport plan.
- `epsilon = 0.5`: mostly diagonal or near-neighbor plan.
- `epsilon = 0.1`: close to hard assignment, but potentially more iterations.

## Output Shape

The script should print a compact comparison like:

```text
Cost matrix:
...

Hungarian/exact baseline:
  permutation: (0, 1, 2, 3)
  cost: ...
  entropy: ...
  plan:
  ...

Sinkhorn epsilon=2.0:
  cost: ...
  entropy: ...
  iterations: ...
  max row error: ...
  max col error: ...
  plan:
  ...
```

## Comparison Points

After running the demo, the explanation should emphasize:

- Hungarian/exact assignment gives a sparse, one-to-one matching.
- Sinkhorn gives a dense transport plan whose softness is controlled by
  `epsilon`.
- As `epsilon` decreases, the Sinkhorn plan moves toward the exact assignment.
- Lower `epsilon` can require more iterations and can become numerically fragile.
- For the small demo, the goal is not speed; it is to show the difference between
  hard matching and entropy-regularized transport.

## Verification

Run:

```bash
uv sync
uv run python sinkhorn_benchmark.py
```

Check:

- Hungarian/exact assignment matrix has one nonzero entry per row and column.
- Sinkhorn row sums are close to `a`.
- Sinkhorn column sums are close to `b`.
- Sinkhorn costs become closer to the exact baseline as `epsilon` decreases.

## Default Benchmark Result

Default command:

```bash
uv run python sinkhorn_benchmark.py
```

On the local run with seed `7`, five random trials per size, and epsilons
`0.5`, `0.1`, and `0.02`, the key `n=200` rows were:

```text
n    exact_ms  exact_cost  eps    sink_ms  sink_cost  abs_err   assign_overlap  assign_err  iters
200  842.157   0.001136    0.5    0.618    0.115116   0.113980  0.006740        0.993260    10
                         0.1    0.762    0.039495   0.038358  0.011580        0.988420    32
                         0.02   2.417    0.009973   0.008836  0.022382        0.977618    152
```

The assignment matrix comparison uses this normalized L1 distance:

```text
A = Hungarian permutation matrix, row/column sums = 1
S = n * Sinkhorn transport plan, row/column sums = 1
assign_err = sum(abs(A - S)) / (2n)
```

So `assign_err = 0` means identical assignments, and `assign_err = 1` means no
mass on the Hungarian assignment. `assign_overlap` is the fraction of Sinkhorn
mass placed on the Hungarian pairs, so `assign_err = 1 - assign_overlap`.

Interpretation:

- Hungarian gives the exact sparse assignment, but took about `842 ms` at
  `n=200` in this pure Python implementation.
- Sinkhorn took about `0.6 ms` to `2.4 ms`, depending on `epsilon`.
- Smaller `epsilon` moved the Sinkhorn cost closer to the exact assignment:
  average gap dropped from about `0.114` to about `0.0088`.
- Smaller `epsilon` also required more iterations: `10` to `152`.
- The Sinkhorn plan is still much softer than the exact one at `n=200`; it spreads
  mass across many columns instead of putting one `1/n` mass on each matched pair.
- The assignment error stays high for these epsilons: even `epsilon=0.02` has
  `assign_err = 0.977618`, meaning it only puts about `2.24%` of its mass on the
  exact Hungarian pairs.

## Implemented Benchmark

The first implementation is:

```text
sinkhorn_benchmark.py
```

It prints:

- one detailed random example with the cost matrix, exact plan, and Sinkhorn
  plans;
- a benchmark table over random point sets;
- exact assignment time;
- Sinkhorn time for several `epsilon` values;
- raw cost gap against exact assignment;
- entropy;
- mass placed on the exact assignment;
- iteration count.

Default command:

```bash
uv run python sinkhorn_benchmark.py
```

Useful variants:

```bash
uv run python sinkhorn_benchmark.py --seed 123
uv run python sinkhorn_benchmark.py --sizes 3 4 5 6 7 8 9 --trials 10
uv run python sinkhorn_benchmark.py --epsilons 1.0 0.2 0.05 0.01
uv run python sinkhorn_benchmark.py --method scaling --sizes 200 --trials 1 --epsilons 0.001 0.0001 0.00001 0.000001 0.0000002 0.0000001 0.00000005 --max-iter 5000
```

The exact baseline is now the Hungarian algorithm, so sizes like `200` are fine.
For much larger sizes, the exact baseline will become the slow part.

## Epsilon Needed For Small Assignment Error

For the `n=200` random draw with seed `7`, plain primal Sinkhorn did not make the
assignment matrix close to Hungarian. It needed log-domain Sinkhorn with epsilon
scaling.

Using:

```bash
uv run python sinkhorn_benchmark.py --method scaling --detail-size 1 --sizes 200 --trials 1 --epsilons 0.001 0.0005 0.0002 0.0001 0.00005 0.00002 0.00001 0.000005 0.000002 0.000001 0.0000005 0.0000002 0.0000001 0.00000005 --max-iter 5000 --tol 1e-8
```

the assignment error was:

```text
epsilon   assignment_error
1e-6      0.159194
5e-7      0.109628
2e-7      0.063089
1e-7      0.035107
5e-8      0.014954
```

The same run had these timings on the `n=200` matrix:

```text
Hungarian exact: 394.804 ms

epsilon   step_time_ms   assignment_error
2e-7      1157.343       0.063089
1e-7      1144.961       0.035107
5e-8       436.876       0.014954
```

Those are per-epsilon warm-started step times. The cumulative time to scale down
from `1e-3` to the target epsilon was much larger:

```text
target epsilon   cumulative_scaling_time
2e-7             about 12.8 s
1e-7             about 13.9 s
5e-8             about 14.3 s
```

So, for this data scale:

- below `0.1` error needs about `epsilon = 2e-7`;
- below `0.05` error needs about `epsilon = 1e-7`;
- below `0.02` error needs about `epsilon = 5e-8`.

This is no longer the fast, simple Sinkhorn regime. It requires log-domain
stabilization and epsilon scaling, and it gives up most of Sinkhorn's speed
advantage if the goal is to recover the hard Hungarian assignment.

## Root Tracking Use Case

For tracking polynomial roots between nearby parameter values, the question is
usually not whether the whole soft Sinkhorn matrix equals the Hungarian
permutation. A better practical check is:

- take the row argmax of the scaled Sinkhorn matrix `S = n * P`;
- verify the argmax columns are unique;
- verify the top mass and top-vs-second margin are high;
- fall back to Hungarian if any row is ambiguous or duplicate columns appear.

The root-specific benchmark is:

```bash
uv run python root_tracking_benchmark.py
```

For a synthetic degree-50 case with small root motion, default seed `123`, and
20 primal Sinkhorn iterations:

```text
Hungarian time: 1.651 ms

scale  epsilon       sink_ms  argmax_hung  unique_cols  assign_err  mean_top  min_top
1      0.00758507    0.208    1.000        50           0.270401    0.7296    0.4438
0.2    0.00151701    0.147    1.000        50           0.069522    0.9305    0.5852
0.02   0.000151701   0.136    1.000        50           0.001863    0.9981    0.9674
0.01   0.0000758507  0.134    1.000        50           0.000570    0.9994    0.9750
```

In that easy, well-separated case, Sinkhorn row argmax recovered the same
assignment as Hungarian and was faster than the pure Python Hungarian
implementation.

For a harder synthetic case with larger root motion:

```bash
uv run python root_tracking_benchmark.py --step 0.08 --iterations 20
```

the table showed duplicate columns and lower agreement for some epsilons. That is
the situation where Sinkhorn should be treated as a proposal or confidence signal,
not a replacement for an exact assignment solver.

Practical recommendation for roots:

- If you need a guaranteed permutation, use Hungarian.
- If roots move in small steps and are well separated, Sinkhorn row argmax can be
  a fast heuristic.
- Use the Sinkhorn top mass and margin as confidence checks.
- If columns are duplicated, top mass is low, or top-vs-second margin is small,
  fall back to Hungarian.
- Near root collisions or branch points, closest-distance matching itself can be
  the wrong notion of "same root"; smaller parameter steps or continuation
  information are more important than the assignment algorithm.

## Optional Follow-Up

If the sketch needs to become slightly more realistic later:

- Add a log-domain Sinkhorn implementation for very small `epsilon`.
- Compare the pure Python Hungarian implementation with SciPy's optimized
  `linear_sum_assignment`.
- Plot the transport plans as heatmaps.
- Try non-uniform marginals.
- Try absolute distance instead of squared distance.
