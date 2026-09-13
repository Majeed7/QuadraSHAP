from __future__ import annotations

import itertools
import math
from typing import Iterable, Sequence

import numpy as np


def naive_shapley(a: Sequence[float]) -> np.ndarray:
    """Naive Shapley values for the product game.

    Game definition
    -------------
    For players 0..d-1 with factors ``a[i]``:

        v(S) = prod_{i in S} a[i] - 1

    So v(∅)=0 and v(M)=prod(a)-1.

    Returns
    -------
    phi : ndarray, shape (d,)
        Shapley values.
    """

    a = np.asarray(a, dtype=np.float64)
    d = int(a.shape[0])
    if d == 0:
        return np.zeros((0,), dtype=np.float64)

    phi = np.zeros((d,), dtype=np.float64)

    # Precompute factorial weights
    fact = [math.factorial(i) for i in range(d + 1)]
    denom = fact[d]

    players = list(range(d))
    for i in range(d):
        others = [p for p in players if p != i]
        for r in range(d):
            # subsets of size r from others
            w = fact[r] * fact[d - r - 1] / denom
            for S in itertools.combinations(others, r):
                prod_S = 1.0
                for j in S:
                    prod_S *= a[j]
                phi[i] += w * (prod_S * (a[i] - 1.0))

    return phi


def naive_shapley_absent(u: Sequence[float], ut: Sequence[float]) -> np.ndarray:
    """Naive Shapley values of the product game with present factors ``u`` and absent factors ``ut``.

        v(S) = prod_{j in S} u[j] * prod_{j not in S} ut[j]

    ``ut = 1`` recovers :func:`naive_shapley` (the constant shift ``-1`` does not
    affect Shapley values).  Exponential in ``d``; for tests only.
    """
    u = np.asarray(u, dtype=np.float64)
    ut = np.asarray(ut, dtype=np.float64)
    d = int(u.shape[0])
    if d == 0:
        return np.zeros((0,), dtype=np.float64)
    fact = [math.factorial(i) for i in range(d + 1)]
    phi = np.zeros(d, dtype=np.float64)
    for i in range(d):
        others = [p for p in range(d) if p != i]
        for k in range(d):
            wgt = fact[k] * fact[d - k - 1] / fact[d]
            for S in itertools.combinations(others, k):
                mask = np.zeros(d, dtype=bool)
                mask[list(S)] = True
                base = np.prod(np.where(mask, u, ut)[others])
                phi[i] += wgt * (u[i] - ut[i]) * base
    return phi
