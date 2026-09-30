//! W4: pc-rmt-spca-dense runs FISTA at lambda = 0 (`--lambda_frac none
//! --lambda_abs 0`). The soft-threshold is then |z| - 0, i.e. the identity, and
//! nothing divides by lambda, so FISTA reduces to orthogonal iteration on
//! I + 2*gamma*S. Check what the dense arm relies on:
//!   * the loadings are dense (no entry zeroed),
//!   * they are orthonormal,
//!   * every eigenvector of the biwhitened covariance S clearly above the MP
//!     edge (> 1.1 lambda+) lies in span(W). The span, not the columns: FISTA
//!     stops on Tr(W^T S W), which is blind to rotations inside the subspace,
//!     and X @ W distances (all kNN sees) are too.
//!     Components AT the edge are not exact, and need not be: their eigengaps
//!     to the bulk are ~1%, orthogonal iteration contracts them at ~0.998 per
//!     step, and FISTA stops on tol_obj first. Measured on this data: k = 9,
//!     the 5 above ~1.07 lambda+ exact, the 4 edge ones 95.5-100% inside, total
//!     deficit 0.049 of 9. The sparse default arm stops by the same rule, so
//!     lambda stays the only difference between the two arms; an exact-EVD
//!     bypass at lambda = 0 would add a second one.
//! and, as a control, that the default lambda_frac = 0.3 does zero entries.
//!
//!     cargo test --release --test lambda_zero

use faer::Mat;
use rmt_spca::spca::{FistaConfig, SparsePCA};

/// Deterministic Poisson counts with `k` planted programmes, log1p'd.
fn synthetic(n: usize, p: usize, k: usize) -> Mat<f64> {
    let mut s: u64 = 0x9E3779B97F4A7C15;
    let mut unif = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let base: Vec<f64> = (0..p).map(|_| 0.5 + 2.0 * unif()).collect();
    let prog: Vec<Vec<bool>> = (0..k).map(|_| (0..p).map(|_| unif() < 0.1).collect()).collect();
    let mut x = Mat::zeros(n, p);
    for i in 0..n {
        let w: Vec<f64> = (0..k).map(|_| unif()).collect();
        for j in 0..p {
            let boost: f64 = (0..k).filter(|&c| prog[c][j]).map(|c| 4.0 * w[c]).sum();
            let rate = base[j] * (1.0 + boost);
            // Knuth: fine for these small rates
            let (l, mut kk, mut prod) = ((-rate).exp(), 0u32, unif());
            while prod > l {
                kk += 1;
                prod *= unif();
            }
            x.as_mut().write(i, j, (kk as f64).ln_1p());
        }
    }
    x
}

fn fit(x: &Mat<f64>, lambda_frac: Option<f64>, lambda: f64) -> rmt_spca::spca::SparsePCAResult {
    SparsePCA::new(FistaConfig {
        lambda_frac,
        lambda,
        k_max: 50,
        top_eigvec_count: 50,
        ..FistaConfig::default()
    })
    .fit(x)
}

fn zero_frac(w: &Mat<f64>) -> f64 {
    let (p, k) = (w.nrows(), w.ncols());
    let z = (0..p).flat_map(|i| (0..k).map(move |j| (i, j))).filter(|&(i, j)| w.read(i, j) == 0.0).count();
    z as f64 / (p * k) as f64
}

#[test]
fn lambda_zero_is_dense_pca_subspace() {
    let x = synthetic(400, 300, 3);
    let res = fit(&x, None, 0.0);
    let w = &res.components;
    let (p, k) = (w.nrows(), w.ncols());
    assert!(k >= 1 && k < res.top_eigenvectors.len(), "k={k}");

    assert_eq!(zero_frac(w), 0.0, "lambda=0 zeroed loadings");

    for a in 0..k {
        for b in 0..k {
            let wtw: f64 = (0..p).map(|i| w.read(i, a) * w.read(i, b)).sum();
            let eye = if a == b { 1.0 } else { 0.0 };
            assert!((wtw - eye).abs() < 1e-8, "W^T W[{a},{b}] = {wtw}");
        }
    }
    // |P_W e|^2 for eigenvector e: 1 iff e lies in span(W)
    let inside = |e: &Vec<f64>| -> f64 {
        (0..k).map(|b| (0..p).map(|i| e[i] * w.read(i, b)).sum::<f64>().powi(2)).sum()
    };
    let ev: Vec<f64> = res.s_eigenvalues.iter().rev().cloned().collect(); // descending
    let clear = ev.iter().take(k).filter(|&&l| l > 1.1 * res.lambda_plus).count();
    assert!(clear >= 1, "no eigenvalue clearly above lambda+ -- test data too weak");
    for a in 0..clear {
        let f = inside(&res.top_eigenvectors[a]);
        assert!(f > 1.0 - 1e-6, "eigenvector {a} (lambda {:.3}) only {f:.6} inside span(W)", ev[a]);
    }
    let deficit: f64 = (0..k).map(|a| 1.0 - inside(&res.top_eigenvectors[a])).sum();
    eprintln!("k={k}  zero frac 0  {clear} clear components exact  total deficit {deficit:.3} of {k}");
}

#[test]
fn default_lambda_is_sparse() {
    let x = synthetic(400, 300, 3);
    let z = zero_frac(&fit(&x, Some(0.3), 0.1).components);
    eprintln!("lambda_frac 0.3: zero frac {z:.3}");
    assert!(z > 0.1, "control: lambda_frac 0.3 should zero entries, got {z}");
}

