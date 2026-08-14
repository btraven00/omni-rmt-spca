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
    return p.parse_args()


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

    # Loud, because it is silent otherwise and it invalidates the run's premise:
    # on non-convergence the crate falls back to per-gene standardisation, so
    # what ran is "standardised sparse PCA with MP thresholding", NOT biwhitened
    # RMT-sPCA. Measured on the be1 fixture this happens in EVERY configuration
    # tried (2000 and 10000 genes, raw counts and log-normalised, bw_damp
    # 1.0/0.8/0.5), residual plateauing at 1.2e-2..3.0e-2.
    if not diag.get("sk_converged", False):
        print(f"  WARNING: biwhitening did NOT converge ({diag['sk_iters']} iters); "
              f"the crate fell back to per-gene standardisation. sigma^2="
              f"{diag['sigma_sq']:.4f} (want ~1.0). Treat this run as the fallback "
              f"method, not as RMT-sPCA.", flush=True)

    cols = [f"PC{i + 1}" for i in range(k)]
    write_tsv(out / f"{args.name}_pcas.tsv", scores, cell_ids, cols, "cell_id")
    write_tsv(out / f"{args.name}_loadings.tsv", W, [gene_ids[j] for j in kept], cols, "gene_id")
    (out / f"{args.name}_rmt.json").write_text(json.dumps(diag, indent=2))
    print(f"  wrote: {out}/{args.name}_{{pcas,loadings}}.tsv + _rmt.json")


if __name__ == "__main__":
    main()
