# omni-rmt-spca (spike)

The omnibenchmark **module**, wrapping RMT-guided sparse PCA —
[btraven00/rmt-spca](https://github.com/btraven00/rmt-spca), implementing
Chardès et al. 2025 (arXiv:2509.15429) — for the `PCA` stage. The `omni-`
prefix marks the benchmark wrapper; `rmt-spca` upstream is the algorithm crate,
and this repo contains no algorithm of its own.

Nothing is packaged. The conda env ships `cargo`; `pca.py` builds the shim in
`rust/` on first use and caches the binary in `rust/target/release`. Python owns
I/O (the stage emits TENx HDF5), Rust owns the arithmetic.

## The module REFUSES to benchmark the fallback

Above a Sinkhorn residual of 1e-2 the crate discards the biwhitening factors and
substitutes per-gene (one-sided) standardisation — a different method from the
one under test. This module therefore **exits non-zero** in that case rather
than emitting an embedding, so a fallback run fails the snakemake job instead of
quietly entering the results. `--max_bw_residual` (default 1e-2, the crate's own
cliff) sets the bar; lower it to demand a better fit.

The bar is "the factors were applied", not "Sinkhorn converged". Convergence
means residual < `tol` = 1e-6, which was **never reached in any run** on real
data — gating on that would fail everything, always.

**Use `--bw_damp 0.3`.** The 1.0 default fails the gate on every dataset tested;
0.3 passes on all of them. Measured:

| dataset | q | damp 1.0 | damp 0.3 |
|---|---|---|---|
| be1 (1715 × 2000) | 1.17 | 3.00e-2 ✗ | 6.88e-3 ✓ |
| pbmc (156881 × 2000) | 0.0127 | 3.34e-2 ✗ | 7.75e-3 ✓ |

Crossing the cliff improves the be1 MP fit: KS 0.0326 → 0.0190. Raising
`--bw_max_iter` does nothing — Sinkhorn exits on stagnation detection (no ≥1%
residual improvement over 100 iterations), not on the cap.

## Scale: it reaches pbmc

Full pbmc (156,881 cells × 2000 genes): **2 min 30 s, 7.6 GB peak**. Cost is
O(genes³) for the EVD and linear in cells, so cells are nearly free. For
contrast, tPCA cannot run this at all (~3.6 TB dense) and the sparse tPCA
rewrite would take ~2 days.

Open caveat: **KS = 0.342 on pbmc vs 0.019 on be1**. The Marchenko-Pastur null
describes be1's bulk well and pbmc's poorly, and λ+ is what selects components.

Where it sits, be1 fixture, matched k=20 — NOTE these numbers predate the gate
and came from the fallback path; they need re-running with `--bw_damp 0.3`:

| k=20 | silhouette ↑ | calinski ↑ | davies-bouldin ↓ |
|---|---|---|---|
| rmt-spca | **0.343** | **489.4** | **1.255** |
| scanpy PCA | 0.327 | 407.8 | 1.292 |
| tPCA | 0.127 | 135.5 | — |

Beating stock PCA on all three while its headline step was disabled is
encouraging, not conclusive.

## k is derived, not requested

k = the number of covariance eigenvalues above the MP bulk edge
λ+ = (1+√q)². `--n_components` is accepted (the stage passes it) but used only
as a **cap**. **Every run so far returns k=20, which is the crate's internal `k_max`** — on
be1, on 10k genes, and on pbmc, across q = 0.0127 to 5.83. On pbmc λ_max = 312
against λ+ = 1.24, so the true count is far higher. The cap binds silently, so
"the method chooses k" has not actually been observed yet. Raise `k_max`
upstream before scheduling a run.

## Diagnostics

`{name}_rmt.json` carries k, λ+, q, σ², KS distance, `sk_iters`,
`sk_converged`. It is **not** a declared stage output — omnibenchmark tracks
only stage-level outputs, so the collector will not see it. It survives in the
run directory.

## Scaling, vs the tPCA modules

Cost is O(n_genes² ) for the covariance and O(p³) for the EVD — **linear in
cells**. Opposite of tPCA/Randomly, which are O(n_cells²). More cells is cheap;
more genes is what hurts (p=10000 EVD is already slow).
