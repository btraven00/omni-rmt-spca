#!/usr/bin/env python3
"""
RMT-guided sparse PCA (Chardès et al. 2025) as an omnibenchmark PCA module.

Wraps github.com/btraven00/rmt-spca. Python owns the I/O (the stage hands us a
TENx HDF5, which the Rust I/O crate cannot read anyway), Rust owns the
arithmetic. Nothing is packaged: the conda env ships `cargo`, and this wrapper
builds the shim in `rust/` on first use and caches the binary in target/.

Pipeline (all in the crate): Sinkhorn-Knopp biwhitening -> mean-centre ->
covariance -> full EVD with Marchenko-Pastur bulk-median sigma^2 and a KS
goodness-of-fit -> subspace iteration above lambda+ -> FISTA sparse PCA.

Outputs
-------
{output_dir}/{name}_pcas.tsv       cell_id  PC1..PCk
{output_dir}/{name}_loadings.tsv   gene_id  PC1..PCk
{output_dir}/{name}_rmt.json       k, lambda_plus, q, sigma_sq, KS, Sinkhorn state

k is NOT --n_components
-----------------------
The method *derives* k: the number of covariance eigenvalues above the MP bulk
edge lambda+ = (1+sqrt(q))^2. That is its whole contribution, so --n_components
is accepted (the stage passes it) and used only as a cap, never as a target.
The k actually found is written to the JSON and is the number of columns in the
TSVs. Compare across modules at your own risk: a natural-k module and a fixed-k
module are not producing the same object.
"""

import argparse
import json
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
RUST = HERE / "rust"
BIN = RUST / "target" / "release" / "omni-rmt-spca"


def parse_args():
    p = argparse.ArgumentParser(description="RMT-guided sparse PCA module")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--normalized_selected_h5", required=True,
                   help="TENx-format HDF5 of normalized, selected expression (genes x cells)")

    p.add_argument("--lambda_frac", default="0.3",
                   help="lambda = F x lambda+; ~0.3 targets 200-500 genes/component. "
                        "'none' falls back to --lambda_abs")
    p.add_argument("--lambda_abs", type=float, default=0.1,
                   help="absolute L1 penalty; ignored unless --lambda_frac none")
    p.add_argument("--eigensolver", choices=["full", "fast"], default="full",
                   help="full = exact O(p^3) EVD + KS diagnostic; fast = approximate")
    p.add_argument("--bw_max_iter", type=int, default=1000)
    p.add_argument("--bw_damp", type=float, default=1.0,
                   help="Sinkhorn under-relaxation; try 0.5-0.8 if biwhitening oscillates")
    p.add_argument("--n_components", type=int, default=0,
                   help="CAP only, not a target: 0 = keep every component RMT finds")
    p.add_argument("--max_bw_residual", type=float, default=1e-2,
                   help="fail the run if the Sinkhorn residual exceeds this. Default 1e-2 "
                        "is the crate's own fallback cliff, so the default behaviour is "
                        "'never benchmark the fallback'. Lower it to demand a better fit")
    return p.parse_args()


def check_biwhitening(diag, max_residual):
    """Refuse to emit an embedding the fallback produced.

    Above the crate's 1e-2 cliff it discards the Sinkhorn factors and uses
    per-gene (one-sided) standardisation -- a different method from the one
    under test, so a benchmark number from it is measuring the wrong thing.
    Exiting non-zero fails the snakemake job rather than quietly publishing it.

    NOTE the bar is 'the factors were applied', not 'Sinkhorn converged'.
    Convergence means residual < tol = 1e-6, which was never reached in any run
    on real data (be1 or pbmc, any damping) -- gating on that would fail
    everything, always.
    """
    res = diag["sk_residual"]
    if res > max_residual:
        sys.exit(
            f"error: biwhitening residual {res:.2e} > --max_bw_residual {max_residual:.0e} "
            f"after {diag['sk_iters']} iters.\n"
            f"       The crate discards the Sinkhorn factors above 1e-2 and falls back to "
            f"per-gene standardisation, which is NOT the method under test.\n"
            f"       Fix: --bw_damp 0.3 (measured: 1.0 -> 3.0e-2 fails, 0.3 -> 6.9e-3 passes "
            f"on both be1 and pbmc). Raising --bw_max_iter does not help: Sinkhorn exits on "
            f"stagnation detection, not on the iteration cap."
        )


def read_tenx_h5(path):
    """TENx genes x cells CSC -> dense cells x genes, plus ids."""
    with h5py.File(path, "r") as h5:
        g = h5["matrix"]
        X = sp.csc_matrix(
            (g["data"][:], g["indices"][:], g["indptr"][:]), shape=tuple(g["shape"][:])
        )
        gene_ids = g["genes"][:].astype(str)
        cell_ids = g["barcodes"][:].astype(str)
    return np.asarray(X.T.todense(), dtype=np.float64), list(cell_ids), list(gene_ids)


def ensure_binary():
    """Build the Rust shim on first use. cargo comes from the conda env."""
    if BIN.exists():
        return BIN
    if shutil.which("cargo") is None:
        sys.exit("error: cargo not found on PATH; the module env must provide rust")
    print("  building rust shim (first run only)...", flush=True)
    subprocess.run(["cargo", "build", "--release"], cwd=RUST, check=True)
    return BIN


def run_rust(X, args, workdir):
    mat = workdir / "matrix.bin"
    with open(mat, "wb") as f:
        f.write(struct.pack("<QQ", *X.shape))
        f.write(np.ascontiguousarray(X, dtype="<f8").tobytes())

    loadings, diag = workdir / "loadings.tsv", workdir / "diag.json"
    subprocess.run(
        [str(ensure_binary()), str(mat), str(loadings), str(diag),
         args.lambda_frac, str(args.lambda_abs), str(args.bw_max_iter),
         str(args.bw_damp), args.eigensolver],
        check=True,
    )
    W = np.atleast_2d(np.loadtxt(loadings))
    return W, json.loads(diag.read_text())


def write_tsv(path, matrix, row_ids, col_names, row_label):
    with open(path, "w") as f:
        f.write(row_label + "\t" + "\t".join(col_names) + "\n")
        for rid, row in zip(row_ids, matrix):
            f.write(rid + "\t" + "\t".join(f"{v:.10g}" for v in row) + "\n")


def main():
    args = parse_args()
    print(f"Full command: {' '.join(sys.argv)}")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    X, cell_ids, gene_ids = read_tenx_h5(args.normalized_selected_h5)
    print(f"  matrix (cells x genes): {X.shape}")

    with tempfile.TemporaryDirectory() as d:
        W, diag = run_rust(X, args, Path(d))

    # Gate before doing anything else: a fallback embedding must not reach the
    # benchmark at all, so fail here rather than after writing the TSVs.
    print(f"  biwhitening: {diag['sk_iters']} iters, residual={diag['sk_residual']:.2e}, "
          f"converged={diag['sk_converged']}, used_fallback={diag['used_fallback']}")
    check_biwhitening(diag, args.max_bw_residual)

    # Stage 0 drops all-zero genes before fitting, so W has one row per
    # SURVIVING gene. Map back by position over the genes we actually sent,
    # in the same order the crate filtered them (column order preserved).
    kept = [j for j in range(X.shape[1]) if np.any(X[:, j] != 0.0)]
    if W.shape[0] != len(kept):
        sys.exit(f"error: loadings rows {W.shape[0]} != surviving genes {len(kept)}")

    if args.n_components and W.shape[1] > args.n_components:
        print(f"  capping k {W.shape[1]} -> {args.n_components}")
        W = W[:, : args.n_components]

    scores = X[:, kept] @ W  # project_cells(): plain X @ W
    k = W.shape[1]
    print(f"  k={k} lambda+={diag['lambda_plus']:.4f} sigma^2={diag['sigma_sq']:.4f} "
          f"KS={diag['ks_distance']}")

    # Three distinct outcomes, and only the first invalidates the run's premise.
    # Keying this on sk_converged alone is WRONG: it is false both when the crate
    # discarded the Sinkhorn factors and when it applied imperfect ones. The
    # fallback test is the crate's own (spca.rs:221): residual > 1e-2.
    #
    # Measured: bw_damp 1.0 lands above the cliff every time (be1 3.0e-2, pbmc
    # 3.34e-2), bw_damp 0.3 below it every time (be1 6.9e-3, pbmc 7.8e-3).
    # Sinkhorn never reaches tol=1e-6 on real data; it exits on stagnation
    # detection (no >=1% improvement over 100 iters), NOT on --bw_max_iter, so
    # raising that flag does nothing.
    if not diag["sk_converged"]:
        print("  note: biwhitening stagnated short of tol=1e-6 but stayed under the "
              "fallback cliff, so the real (imperfect) factors were applied.")

    cols = [f"PC{i + 1}" for i in range(k)]
    write_tsv(out / f"{args.name}_pcas.tsv", scores, cell_ids, cols, "cell_id")
    write_tsv(out / f"{args.name}_loadings.tsv", W, [gene_ids[j] for j in kept], cols, "gene_id")
    (out / f"{args.name}_rmt.json").write_text(json.dumps(diag, indent=2))
    print(f"  wrote: {out}/{args.name}_{{pcas,loadings}}.tsv + _rmt.json")


if __name__ == "__main__":
    main()
