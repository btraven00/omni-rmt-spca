# omni-rmt-spca (spike)

The omnibenchmark **module**, wrapping RMT-guided sparse PCA —
[btraven00/rmt-spca](https://github.com/btraven00/rmt-spca), implementing
Chardès et al. 2025 (arXiv:2509.15429) — for the `PCA` stage. The `omni-`
prefix marks the benchmark wrapper; `rmt-spca` upstream is the algorithm crate,
and this repo contains no algorithm of its own.

Nothing is packaged. The conda env ships `cargo`; `pca.py` builds the shim in
`rust/` on first use and caches the binary in `rust/target/release`. Python owns
I/O (the stage emits TENx HDF5), Rust owns the arithmetic.

## Status: spike. Read this before trusting a number.

**Biwhitening never converged on the be1 fixture** — in any configuration tried:
2000 HVGs and 10000 genes, raw counts and log-normalised, `bw_damp` 1.0 / 0.8 /
0.5. The residual plateaus at 1.2e-2 – 3.0e-2 and the crate falls back to
per-gene standardisation (`sk_converged: false`, σ² 0.43–0.61 where ~1.0 is
wanted). So every result below is the **fallback** path: standardised sparse PCA
with MP thresholding, not biwhitened RMT-sPCA. The module now prints a loud
WARNING when this happens.

Where it sits anyway, be1 fixture, matched k=20 (its own RMT-chosen k):

| k=20 | silhouette ↑ | calinski ↑ | davies-bouldin ↓ |
|---|---|---|---|
| rmt-spca | **0.343** | **489.4** | **1.255** |
| scanpy PCA | 0.327 | 407.8 | 1.292 |
| tPCA | 0.127 | 135.5 | — |

Beating stock PCA on all three while its headline step is disabled is
encouraging, not conclusive.

## k is derived, not requested

k = the number of covariance eigenvalues above the MP bulk edge
λ+ = (1+√q)². `--n_components` is accepted (the stage passes it) but used only
as a **cap**. On be1 it returns k=20 — which is also the crate's internal
`k_max`, so that value is a ceiling artifact, not necessarily the RMT count.

## Diagnostics

`{name}_rmt.json` carries k, λ+, q, σ², KS distance, `sk_iters`,
`sk_converged`. It is **not** a declared stage output — omnibenchmark tracks
only stage-level outputs, so the collector will not see it. It survives in the
run directory.

## Scaling, vs the tPCA modules

Cost is O(n_genes² ) for the covariance and O(p³) for the EVD — **linear in
cells**. Opposite of tPCA/Randomly, which are O(n_cells²). More cells is cheap;
more genes is what hurts (p=10000 EVD is already slow).
