//! Thin shim: raw matrix in, sparse loadings + RMT diagnostics out.
//!
//! Deliberately NOT using rmt-spca-io. That crate reads .h5ad and pulls in
//! pyo3 (auto-initialize, embeds a Python interpreter) and plotters for the
//! UMAP path -- neither of which belongs inside an omnibenchmark conda env.
//! The stage already hands us a TENx HDF5, and the Python wrapper already has
//! a reader for it, so I/O stays in Python and only the arithmetic is Rust.
//!
//! Wire format (little-endian, written by pca.py):
//!   [n: u64][p: u64][data: f64 * n*p, row-major, cells x genes]
//!
//! Argv: <matrix.bin> <loadings.tsv> <diagnostics.json>
//!       <lambda_frac|none> <lambda> <bw_max_iter> <bw_damp> <full|fast> <k_max>

use std::env;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

use faer::Mat;
use rmt_spca::spca::{EigensolverMode, FistaConfig, SparsePCA};

fn read_matrix(path: &str) -> std::io::Result<Mat<f64>> {
    let mut r = BufReader::new(File::open(path)?);
    let mut u = [0u8; 8];
    r.read_exact(&mut u)?;
    let n = u64::from_le_bytes(u) as usize;
    r.read_exact(&mut u)?;
    let p = u64::from_le_bytes(u) as usize;

    let mut buf = vec![0u8; n * p * 8];
    r.read_exact(&mut buf)?;
    Ok(Mat::from_fn(n, p, |i, j| {
        let o = (i * p + j) * 8;
        f64::from_le_bytes(buf[o..o + 8].try_into().unwrap())
    }))
}

fn main() -> std::io::Result<()> {
    let a: Vec<String> = env::args().collect();
    if a.len() < 10 {
        eprintln!("usage: {} <matrix.bin> <loadings.tsv> <diag.json> <lambda_frac|none> \
                   <lambda> <bw_max_iter> <bw_damp> <full|fast> <k_max>", a[0]);
        std::process::exit(2);
    }

    let k_max_req: usize = a[9].parse().unwrap();
    let data = read_matrix(&a[1])?;
    eprintln!("  matrix (cells x genes): {} x {}", data.nrows(), data.ncols());

    let config = FistaConfig {
        lambda_frac: if a[4] == "none" { None } else { Some(a[4].parse().unwrap()) },
        lambda: a[5].parse().unwrap(),
        bw_max_iter: a[6].parse().unwrap(),
        bw_damp: a[7].parse().unwrap(),
        eigensolver: if a[8] == "fast" { EigensolverMode::Fast } else { EigensolverMode::Full },
        // KS only exists in Full mode; asking for it in Fast is a silent no-op.
        compute_ks: a[8] != "fast",
        k_max: k_max_req,
        verbose: true,
        ..FistaConfig::default()
    };

    let k_max_eff = k_max_req.min(data.ncols()).min(data.nrows());
    let res = SparsePCA::new(config).fit(&data);
    let (p_out, k) = (res.components.nrows(), res.components.ncols());

    // The k the crate returns is min(true k, internal k_max). k_max is a
    // hard-coded 20 (spca.rs:364) and the subspace iteration never looks past
    // it, so a capped run is indistinguishable from a genuine k=20 -- silently.
    // In Full mode we have the whole rescaled spectrum, so count the real
    // number of signal eigenvalues and report both. k_rmt_true == -1 means Fast
    // mode, where s_eigenvalues is empty and the count is unavailable.
    let k_rmt_true: i64 = if res.s_eigenvalues.is_empty() {
        -1
    } else {
        res.s_eigenvalues.iter().filter(|&&e| e > res.lambda_plus).count() as i64
    };
    eprintln!("  k returned: {k}   k above lambda+ (uncapped): {k_rmt_true}");

    // Stage 0 drops all-zero genes, so p_out can be < the p we sent. The
    // wrapper needs to know: it maps rows back to gene ids by position, and a
    // silent shortfall would misalign every loading.
    let mut w = BufWriter::new(File::create(&a[2])?);
    for i in 0..p_out {
        let row: Vec<String> = (0..k).map(|c| format!("{:.10}", res.components.read(i, c))).collect();
        writeln!(w, "{}", row.join("\t"))?;
    }
    w.flush()?;

    let mut j = BufWriter::new(File::create(&a[3])?);
    write!(
        j,
        // sk_residual is what actually decides the fallback (spca.rs:221 --
        // `!bw_ok && bw_res > 1e-2`). sk_converged alone cannot tell "applied
        // imperfect factors" from "discarded them": it is false in both cases,
        // and those are completely different runs.
        "{{\"k\":{},\"p_out\":{},\"lambda_plus\":{},\"q\":{},\"sigma_sq\":{},\
         \"ks_distance\":{},\"sk_iters\":{},\"sk_converged\":{},\"sk_residual\":{},\
         \"used_fallback\":{},\"k_rmt_true\":{},\"k_max\":{},\"k_capped\":{},\
         \"eigenvalues\":{:?}}}",
        k, p_out, res.lambda_plus, res.q, res.sigma_sq,
        res.ks_distance.map(|v| v.to_string()).unwrap_or_else(|| "null".into()),
        res.sk_iters, res.sk_converged, res.sk_residual,
        !res.sk_converged && res.sk_residual > 1e-2,
        k_rmt_true, k_max_eff,
        // The cap binds iff k saturates it. NOT `k_rmt_true > k`: the two counts
        // come from different estimators (subspace-iteration Rayleigh quotients
        // vs the full EVD spectrum) and disagree by ~1 at the boundary, where an
        // eigenvalue sits essentially on lambda+. pbmc: 393 vs 394.
        k >= k_max_eff,
        res.eigenvalues
    )?;
    j.flush()?;
    Ok(())
}
