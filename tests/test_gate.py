#!/usr/bin/env python3
"""Self-check for the biwhitening gate: `python tests/test_gate.py`.

The gate is the one piece of logic that decides whether a number reaches the
benchmark, so it gets the one test. No fixtures, no rust binary needed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pca import check_biwhitening  # noqa: E402


def diag(residual, converged=False, iters=100):
    return {"sk_residual": residual, "sk_converged": converged, "sk_iters": iters}


def rejects(d, limit):
    try:
        check_biwhitening(d, limit)
    except SystemExit:
        return True
    return False


# Measured values: bw_damp 1.0 lands above the cliff on both datasets, 0.3 below.
assert rejects(diag(3.00e-2), 1e-2), "be1 @ damp 1.0 must fail"
assert rejects(diag(3.34e-2), 1e-2), "pbmc @ damp 1.0 must fail"
assert not rejects(diag(6.88e-3), 1e-2), "be1 @ damp 0.3 must pass"
assert not rejects(diag(7.75e-3), 1e-2), "pbmc @ damp 0.3 must pass"

# The boundary is the crate's own, and it is exclusive: 1e-2 exactly is applied,
# not discarded (spca.rs:221 tests `> 1e-2`).
assert not rejects(diag(1e-2), 1e-2), "exactly at the cliff must pass"
assert rejects(diag(1.0001e-2), 1e-2), "just above the cliff must fail"

# Tightening the bar must reject runs the default accepts.
assert rejects(diag(6.88e-3), 1e-3), "stricter limit must reject a passing default"

# Convergence is NOT the bar: tol=1e-6 was never reached on real data, so
# gating on sk_converged would fail every run forever.
assert not rejects(diag(7.75e-3, converged=False), 1e-2), "non-convergence alone must not fail"

print("ok  biwhitening gate: fallback rejected, applied-factors accepted")
