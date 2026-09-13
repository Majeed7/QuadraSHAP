"""
Blockwise evaluation of weighted sums of product games.

The Shapley values of a value function of Section 4 are a sum over component--
background pairs and over quadrature nodes, and both sums are reductions.  The
pairs can therefore be processed in blocks of ``block_size`` and the nodes in
blocks of ``node_block``, each block being accumulated into the running
attribution vector, which bounds peak memory at ``O(block_size * node_block * d)``
without any approximation.  This module decides those block sizes from the
memory that is actually available on the machine (or from an explicit budget),
so that blocking is only used when the full computation would not fit, and it
pads partial blocks so that JIT-compiled backends see a single shape.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
import subprocess
import sys
from typing import Optional, Tuple, Union

import numpy as np

BYTES = 8  # float64 working precision of the NumPy backends

# Rough number of ``(node_block, block_size, d)`` float64 tensors alive at the peak of each core,
# plus, for the log-space paths, the cost that does not scale with the node axis.
_TENSORS_PER_BACKEND = {
    "prefix_scan_numpy": 4.0,   # B, pref, suf (pref*suf in place) and the reduction temporary
    "prefix_scan_jax": 4.0,
    "logspace_numpy": 0.0,      # keeps only (node_block, block_size) accumulators
    "logspace_jax": 2.0,        # vmap over features may materialise (d, node_block, block_size)
}
_PER_PAIR_BYTES_FACTOR = 6  # K, Ut, U, Phi and two temporaries of shape (block_size, d)

# Performance cap for the automatic plan.  The prefix-scan core sweeps a (node_block, block_size, d)
# tensor several times per node block (build, prefix, suffix, reduce), so it runs 1.5-2.5x faster
# when one such tensor stays resident in the last-level cache; the log-space core keeps only
# (node_block, block_size) accumulators and can only lose time by being blocked, so it gets no cap.
# ``"cache"`` resolves to the last-level cache of the machine at planning time.
DEFAULT_TARGET_BYTES = {"prefix_scan_numpy": "cache", "prefix_scan_jax": None,
                        "logspace_numpy": None, "logspace_jax": None}

# When the soft target would leave fewer than MIN_EFFICIENT_PAIRS pairs per core call -- which
# happens for a large node count, where a single pair already fills the working set -- the automatic
# plan blocks the quadrature nodes instead and widens the pair blocks to EFFICIENT_BLOCK, since
# thousands of one-pair calls cost more time than the memory they save.
MIN_EFFICIENT_PAIRS = 8
EFFICIENT_BLOCK = 32
# ``estimate_peak_bytes`` counts the ~4 tensors the prefix scan keeps alive, so a peak of
# TARGET_SLACK targets is one tensor of one target: that is the point where a single sweep stops
# fitting the cache and blocking starts paying for its per-call overhead.
TARGET_SLACK = 4
# Used when the machine does not expose its cache topology (a VM or a container).  32 MB is the
# order of a current desktop or server last-level cache and of the Apple-silicon system cache; the
# cost of getting it wrong is bounded -- a slightly early or slightly late switch to blocking.
FALLBACK_CACHE_BYTES = 32 * 2 ** 20


# ----------------------------------------------------------------------------- hardware
def available_memory_bytes(backend: str = "logspace_numpy") -> int:
    """Memory available right now on the device the backend runs on, in bytes.

    Host memory is read from :mod:`psutil` when installed, else from ``sysconf``
    (Linux) or ``sysctl hw.memsize`` (macOS, total rather than available), with a
    4 GB fallback.  For a JAX backend on an accelerator the device's free memory
    is used when the runtime reports it.
    """
    if backend.endswith("_jax"):
        try:  # pragma: no cover - depends on the installed accelerator runtime
            import jax
            dev = jax.devices()[0]
            if dev.platform != "cpu":
                stats = dev.memory_stats() or {}
                limit = stats.get("bytes_limit") or stats.get("bytes_reservable_limit")
                if limit:
                    return int(limit - stats.get("bytes_in_use", 0))
        except Exception:
            pass
    try:
        import psutil  # type: ignore
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        pass
    if sys.platform == "darwin":  # macOS reports total physical memory only; take half as "available"
        try:
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], timeout=2,
                                               stderr=subprocess.DEVNULL)) // 2
        except Exception:
            pass
    return 4 * 1024 ** 3


@lru_cache(maxsize=1)
def last_level_cache_bytes() -> int:
    """Size of the last-level data cache, in bytes (``FALLBACK_CACHE_BYTES`` when unknown).

    Read from ``/sys/devices/system/cpu/cpu0/cache`` on Linux and from ``sysctl`` on macOS
    (``hw.l3cachesize``, falling back to the performance cluster's L2 on Apple silicon, which
    is that machine's last level of per-core cache).  Inside a container or a VM that does not
    expose the host topology the fallback applies, which is the right order of magnitude for a
    current server or laptop and is in any case only a performance heuristic.
    """
    best = 0
    try:
        base = "/sys/devices/system/cpu/cpu0/cache"
        for entry in sorted(os.listdir(base)):
            with open(f"{base}/{entry}/size") as fh:
                s = fh.read().strip().upper()
            mult = {"K": 2 ** 10, "M": 2 ** 20, "G": 2 ** 30}.get(s[-1:], 1)
            best = max(best, int(float(s.rstrip("KMGB")) * mult))
    except Exception:
        pass
    if not best and sys.platform == "darwin":
        for key in ("hw.l3cachesize", "hw.perflevel0.l2cachesize", "hw.l2cachesize"):
            try:
                best = int(subprocess.check_output(["sysctl", "-n", key], timeout=2,
                                                   stderr=subprocess.DEVNULL))
                if best:
                    break
            except Exception:
                continue
    return int(best) if best else FALLBACK_CACHE_BYTES


# ----------------------------------------------------------------------------- planning
@dataclass
class BlockPlan:
    """How one ``explain`` call is split. ``block_size`` pairs and ``node_block`` nodes per core call."""

    backend: str
    n_pairs: int
    m_q: int
    d: int
    block_size: int          # pairs per block (== n_pairs when blocking is off)
    node_block: int          # quadrature nodes per core call (== m_q when not needed)
    n_blocks: int
    budget_bytes: Optional[int]   # None when blocking is switched off
    estimated_peak_bytes: int     # of the chosen plan
    full_peak_bytes: int          # of the unblocked computation
    mode: str                # "off", "fits", "tuned" (blocked for speed), "blocked" (for memory), "fixed"

    @property
    def blocked(self) -> bool:
        return self.n_blocks > 1 or self.node_block < self.m_q

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        mb = lambda b: f"{b / 2**20:.1f} MB" if b >= 2**20 else f"{b / 2**10:.0f} KB"
        budget = "off" if self.budget_bytes is None else mb(self.budget_bytes)
        return (f"BlockPlan({self.mode}: {self.n_pairs} pairs x {self.m_q} nodes x {self.d} features on {self.backend}; "
                f"{self.n_blocks} block(s) of {self.block_size} pairs, {self.node_block} nodes per call; "
                f"peak ~{mb(self.estimated_peak_bytes)} (unblocked ~{mb(self.full_peak_bytes)}), budget {budget})")


def estimate_peak_bytes(backend: str, block_size: int, node_block: int, d: int) -> int:
    """Rough peak working memory of one core call on a block."""
    t = _TENSORS_PER_BACKEND.get(backend, 4.0)
    tensor = t * node_block * block_size * d
    pairs = _PER_PAIR_BYTES_FACTOR * block_size * d
    accum = 3 * node_block * block_size  # log-space accumulators / per-feature temporaries
    return int(BYTES * (tensor + pairs + accum))


def plan_blocks(n_pairs: int, m_q: int, d: int, backend: str, block_size: Union[int, str, None] = "auto",
                memory_budget: Union[int, float, str, None] = None, memory_fraction: float = 0.5,
                target_bytes: Union[int, float, str, None] = "auto", min_block: int = 1) -> BlockPlan:
    """Choose ``block_size`` and ``node_block``.

    Parameters
    ----------
    block_size : ``"auto"`` (default) decides from the memory budget and the performance cap and
        does not block when the full computation is below both; a positive integer fixes the
        number of pairs per block (nodes are then blocked only if even one such block does not
        fit the memory budget); ``-1``, ``0`` or ``None`` switch blocking off entirely.
    memory_budget : hard limit: bytes (int/float), a string such as ``"512MB"``/``"2GB"``, or
        ``None`` for ``memory_fraction`` of the memory currently available on the backend's device.
    target_bytes : soft, performance-motivated cap on the working set of one core call, used only
        in ``"auto"`` mode; ``"auto"`` takes ``DEFAULT_TARGET_BYTES[backend]`` (16 MB for the NumPy
        prefix scan, none for the other backends), ``None`` disables the cap.
    """
    n_pairs = int(max(n_pairs, 1)); m_q = int(max(m_q, 1)); d = int(max(d, 1))
    full = estimate_peak_bytes(backend, n_pairs, m_q, d)

    off = block_size is None or (isinstance(block_size, (int, np.integer)) and block_size <= 0)
    if off:
        return BlockPlan(backend, n_pairs, m_q, d, n_pairs, m_q, 1, None, full, full, "off")

    budget = _parse_budget(memory_budget, backend, memory_fraction)

    if isinstance(block_size, (int, np.integer)):  # fixed number of pairs per block
        bs = int(min(block_size, n_pairs))
        nb = _node_block_for(backend, bs, m_q, d, budget)
        n_blocks = -(-n_pairs // bs)
        return BlockPlan(backend, n_pairs, m_q, d, bs, nb, n_blocks, budget,
                         estimate_peak_bytes(backend, bs, nb, d), full, "fixed")

    if block_size != "auto":
        raise ValueError("block_size must be 'auto', a positive int, or -1/None to switch blocking off")
    target = DEFAULT_TARGET_BYTES.get(backend) if target_bytes == "auto" else target_bytes
    if target == "cache":
        target = last_level_cache_bytes()
    elif target is not None:
        target = _parse_budget(target, backend, fraction=1.0)
    # The soft target is a cache-residency heuristic, so it is applied only when the working set is
    # well past it: blocking a computation that is merely somewhat larger than cache costs more in
    # per-call overhead than it recovers in locality.
    limit = budget if target is None or full <= TARGET_SLACK * target else min(budget, int(target))
    if full <= limit:
        return BlockPlan(backend, n_pairs, m_q, d, n_pairs, m_q, 1, budget, full, full, "fits")
    mode = "blocked" if full > budget else "tuned"

    # Largest block of pairs that fits with all nodes at once. When that would leave fewer than
    # EFFICIENT_BLOCK pairs per call -- which happens for a large node count, where a single pair
    # already fills the working set -- block the *nodes* instead and keep the pair blocks wide
    # enough to amortise the per-call overhead; falling back to one pair per block would turn the
    # reduction into thousands of tiny calls and cost more time than it saves.
    bs = _pairs_for(backend, m_q, d, limit)
    if bs >= min(MIN_EFFICIENT_PAIRS, n_pairs):
        nb = m_q
    else:
        bs_eff = int(min(max(EFFICIENT_BLOCK, min_block), n_pairs))
        nb_eff = _node_block_for(backend, bs_eff, m_q, d, limit)
        if estimate_peak_bytes(backend, bs_eff, nb_eff, d) <= budget:   # the hard budget still rules
            bs, nb = bs_eff, nb_eff
        else:                                                            # too little memory for wide blocks
            bs = max(bs, min_block)
            nb = _node_block_for(backend, bs, m_q, d, budget)
    bs = int(min(max(bs, 1), n_pairs))
    n_blocks = -(-n_pairs // bs)
    return BlockPlan(backend, n_pairs, m_q, d, bs, nb, n_blocks, budget,
                     estimate_peak_bytes(backend, bs, nb, d), full, mode)


def _parse_budget(memory_budget, backend: str, fraction: float) -> int:
    if memory_budget is None:
        return int(max(fraction, 0.01) * available_memory_bytes(backend))
    if isinstance(memory_budget, str):
        s = memory_budget.strip().upper().replace("IB", "B")
        units = {"KB": 2 ** 10, "MB": 2 ** 20, "GB": 2 ** 30, "TB": 2 ** 40, "B": 1}
        for u, mult in units.items():
            if s.endswith(u):
                return int(float(s[: -len(u)]) * mult)
        return int(float(s))
    return int(memory_budget)


def _pairs_for(backend: str, node_block: int, d: int, budget: int) -> int:
    """Largest block_size with estimate_peak_bytes <= budget (all else fixed)."""
    t = _TENSORS_PER_BACKEND.get(backend, 4.0)
    per_pair = BYTES * (t * node_block * d + _PER_PAIR_BYTES_FACTOR * d + 3 * node_block)
    return int(budget // max(per_pair, 1.0))


def _node_block_for(backend: str, block_size: int, m_q: int, d: int, budget: int) -> int:
    """Largest node_block <= m_q that fits for a fixed block_size (at least 1)."""
    t = _TENSORS_PER_BACKEND.get(backend, 4.0)
    fixed = BYTES * _PER_PAIR_BYTES_FACTOR * block_size * d
    per_node = BYTES * (t * block_size * d + 3 * block_size)
    if t == 0.0:  # log-space NumPy: the node axis costs almost nothing
        return m_q
    nb = int((budget - fixed) // max(per_node, 1.0))
    return int(min(max(nb, 1), m_q))


# ----------------------------------------------------------------------------- padding for JIT backends
def pad_block(K: np.ndarray, Ut: Optional[np.ndarray], w: np.ndarray, block_size: int
              ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Pad a partial block to ``block_size`` pairs with rows that contribute exactly nothing.

    Padded rows have ``K = 0`` (so their Phi row is identically zero), absent factors
    ``1`` (so no logarithm of zero is taken) and weight ``0``.
    """
    m = K.shape[0]
    if m >= block_size:
        return K, Ut, w
    pad = block_size - m
    K_p = np.concatenate([K, np.zeros((pad, K.shape[1]), dtype=K.dtype)], axis=0)
    w_p = np.concatenate([np.asarray(w, dtype=np.float64), np.zeros(pad)], axis=0)
    if Ut is None or np.asarray(Ut).shape[0] == 1:
        Ut_p = Ut
    else:
        Ut_p = np.concatenate([Ut, np.ones((pad, K.shape[1]), dtype=Ut.dtype)], axis=0)
    return K_p, Ut_p, w_p
