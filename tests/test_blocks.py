"""Blockwise evaluation over component--background pairs and quadrature nodes (exact, memory-bounded).

Runs under pytest or directly (``python tests/test_blocks.py``).
"""
import sys
import tracemalloc
from pathlib import Path

import numpy as np
from sklearn.kernel_ridge import KernelRidge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quadrashap import QuadraSHAP
from quadrashap.product_games.shapley import ProductGamesShapleyNumpy
from quadrashap.product_games.blocks import (plan_blocks, pad_block, estimate_peak_bytes, available_memory_bytes,
                                             _parse_budget, DEFAULT_TARGET_BYTES, last_level_cache_bytes)


def _model(p=200, d=60, gamma=0.05, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((p, d)); y = X[:, 0] * X[:, 1] + 0.1 * rng.standard_normal(p)
    return KernelRidge(kernel="rbf", gamma=gamma, alpha=0.1).fit(X, y), X


def test_node_blocking_in_prefix_scan_is_exact():
    rng = np.random.default_rng(0); core = ProductGamesShapleyNumpy()
    K = rng.uniform(-1, 1, (6, 30)); Ut = rng.uniform(0, 2, (6, 30))
    for m_q in (1, 7, 16):
        ref = core.phi_matrix_prefix_scan(K, m_q, Ut=Ut)
        for nb in (1, 2, 5, m_q, 100):
            np.testing.assert_allclose(core.phi_matrix_prefix_scan(K, m_q, Ut=Ut, node_block=nb), ref, rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(core.phi_matrix_prefix_scan(K, m_q, node_block=3), core.phi_matrix_prefix_scan(K, m_q), rtol=1e-12, atol=1e-13)


def test_pair_blocking_is_exact_for_every_value_function_and_backend():
    model, X = _model(); x = X[0]; bg = X[1:6]
    for backend in ("prefix_scan_numpy", "logspace_numpy"):
        ex = QuadraSHAP(model, backend=backend, background=bg)
        for vf, kw in (("neutral", {}), ("baseline", {"baseline": X[7]}), ("interventional", {})):
            ref = ex.explain(x, vf, m_q=9, block_size=-1, **kw)
            assert ex.last_block_plan.mode == "off" and ex.last_block_plan.n_blocks == 1
            for bs in (1, 3, 17, 64, 10_000):
                phi = ex.explain(x, vf, m_q=9, block_size=bs, **kw)
                plan = ex.last_block_plan
                assert plan.mode == "fixed" and plan.block_size == min(bs, plan.n_pairs)
                assert plan.n_blocks == -(-plan.n_pairs // plan.block_size)
                np.testing.assert_allclose(phi, ref, rtol=1e-12, atol=1e-12, err_msg=f"{backend} {vf} bs={bs}")
            # blocks straddle background rows: the number of pairs is p * n_b
            assert ex.n_pairs(vf, **kw) == (ex.n if vf == "neutral" else ex.n * (1 if vf == "baseline" else len(bg)))
        # auto plan with a tiny hard budget forces blocking (and node blocking for the scan) and stays exact
        ex_small = QuadraSHAP(model, backend=backend, background=bg, memory_budget="64KB")
        for vf, kw in (("neutral", {}), ("interventional", {})):
            phi = ex_small.explain(x, vf, m_q=9, **kw); plan = ex_small.last_block_plan
            assert plan.mode == "blocked" and plan.n_blocks > 1
            assert plan.estimated_peak_bytes <= 64 * 2 ** 10 * 1.05 or (plan.block_size == 1 and plan.node_block == 1)
            np.testing.assert_allclose(phi, ex.explain(x, vf, m_q=9, block_size=-1, **kw), atol=1e-12)


def test_plan_modes_and_budget_parsing():
    assert _parse_budget("512MB", "logspace_numpy", 0.5) == 512 * 2 ** 20
    assert _parse_budget("2GB", "logspace_numpy", 0.5) == 2 * 2 ** 30
    assert _parse_budget("3000", "logspace_numpy", 0.5) == 3000 and _parse_budget(1234.0, "x", 0.5) == 1234
    assert available_memory_bytes() > 0
    # off
    for off in (-1, 0, None):
        p = plan_blocks(1000, 32, 500, "prefix_scan_numpy", block_size=off)
        assert p.mode == "off" and p.n_blocks == 1 and p.block_size == 1000 and p.node_block == 32 and p.budget_bytes is None
    # fits: everything below budget and below the performance cap
    p = plan_blocks(10, 4, 20, "prefix_scan_numpy", memory_budget="1GB")
    assert p.mode == "fits" and p.n_blocks == 1 and not p.blocked
    # tuned: fits the memory budget but exceeds the cache-size cap of the NumPy scan.  The cap is
    # the machine's last-level cache by default, so it is pinned here to keep the test portable.
    p = plan_blocks(1000, 32, 500, "prefix_scan_numpy", memory_budget="100GB", target_bytes="16MB")
    assert p.mode == "tuned" and p.n_blocks > 1 and p.node_block == 32
    assert p.estimated_peak_bytes <= 16 * 2 ** 20 * 1.05
    assert plan_blocks(1000, 32, 500, "prefix_scan_numpy", memory_budget="100GB", target_bytes=None).mode == "fits"
    # the default cap resolves to a real cache size, and only the scan backends have one
    assert DEFAULT_TARGET_BYTES["prefix_scan_numpy"] == "cache" and DEFAULT_TARGET_BYTES["logspace_numpy"] is None
    assert 2 ** 18 <= last_level_cache_bytes() <= 2 ** 32
    # below the cap, nothing is blocked for speed; well above it, something is
    small = plan_blocks(4, 4, 100, "prefix_scan_numpy", memory_budget="100GB")
    big = plan_blocks(4000, 64, 5000, "prefix_scan_numpy", memory_budget="100GB")
    assert small.mode == "fits" and big.mode == "tuned"
    assert plan_blocks(1000, 32, 500, "logspace_numpy", memory_budget="100GB").mode == "fits"  # no cap for log-space
    # blocked: hard memory budget binding
    p = plan_blocks(1000, 32, 500, "prefix_scan_numpy", memory_budget="8MB", target_bytes="16MB")
    assert p.mode == "blocked" and p.estimated_peak_bytes <= 8 * 2 ** 20 * 1.05
    # even one pair does not fit with all nodes -> node blocking
    p = plan_blocks(1000, 64, 5000, "prefix_scan_numpy", memory_budget="1MB")
    assert p.block_size == 1 and 1 <= p.node_block < 64 and p.estimated_peak_bytes <= 1.5 * 2 ** 20
    # fixed
    p = plan_blocks(1000, 32, 500, "logspace_numpy", block_size=64, memory_budget="1GB")
    assert p.mode == "fixed" and p.block_size == 64 and p.n_blocks == 16
    # estimates are monotone in the block sizes
    assert estimate_peak_bytes("prefix_scan_numpy", 10, 4, 100) < estimate_peak_bytes("prefix_scan_numpy", 20, 4, 100)
    assert estimate_peak_bytes("prefix_scan_numpy", 10, 4, 100) < estimate_peak_bytes("prefix_scan_numpy", 10, 8, 100)


def test_pad_block_contributes_nothing():
    rng = np.random.default_rng(1); core = ProductGamesShapleyNumpy()
    K = rng.uniform(-1, 1, (5, 9)); Ut = rng.uniform(0.2, 2, (5, 9)); w = rng.standard_normal(5)
    Kp, Utp, wp = pad_block(K, Ut, w, 8)
    assert Kp.shape == (8, 9) and Utp.shape == (8, 9) and wp.shape == (8,) and wp[5:].sum() == 0
    Kn, Utn, wn = pad_block(K, None, w, 8)
    assert Utn is None and Kn.shape == (8, 9)
    assert pad_block(K, Ut, w, 5)[0] is K  # no padding needed
    for fn in (core.phi_matrix_prefix_scan, core.phi_matrix_logspace):
        ref = (fn(K, 5, Ut=Ut) * w[:, None]).sum(0)
        np.testing.assert_allclose((fn(Kp, 5, Ut=Utp) * wp[:, None]).sum(0), ref, atol=1e-13)
        np.testing.assert_allclose((fn(Kn, 5, Ut=Utn) * wn[:, None]).sum(0), (fn(K, 5) * w[:, None]).sum(0), atol=1e-13)


def test_peak_memory_is_bounded_by_the_budget():
    model, X = _model(p=300, d=300, gamma=0.01); x = X[0]

    def peak(ex, **kw):
        tracemalloc.start(); phi = ex.explain(x, "neutral", m_q=16, **kw)
        _, pk = tracemalloc.get_traced_memory(); tracemalloc.stop(); return phi, pk

    ex = QuadraSHAP(model, backend="prefix_scan_numpy")
    ref, pk_off = peak(ex, block_size=-1)                    # ~4 * 16 * 300 * 300 * 8 B ~ 46 MB
    assert pk_off > 30 * 2 ** 20
    ex_b = QuadraSHAP(model, backend="prefix_scan_numpy", memory_budget="4MB")
    phi, pk_b = peak(ex_b)
    assert ex_b.last_block_plan.mode == "blocked"
    assert pk_b < 0.3 * pk_off and pk_b < 8 * 2 ** 20
    np.testing.assert_allclose(phi, ref, atol=1e-12)


if __name__ == "__main__":
    for fn_name, fn in list(globals().items()):
        if fn_name.startswith("test_") and callable(fn):
            fn(); print("ok", fn_name)
