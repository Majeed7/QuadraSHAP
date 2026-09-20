"""Independent, stable float64 quadrature reference for positive Cox games.

This uses log1p/expm1 on log-factor differences; it does not form long prefix
products or reuse the JAX kernel being evaluated. It is still quadrature, not
exhaustive coalition enumeration.
"""
import numpy as np
from scipy.special import roots_legendre


def reference_cox(beta, X, background, m_q=64, node_block=8, progress=None):
    beta = np.asarray(beta, dtype=np.float64)
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    background = np.asarray(background)
    nodes, weights = roots_legendre(m_q)
    nodes, weights = (nodes + 1) / 2, weights / 2
    phi = np.zeros_like(X)
    for k, z in enumerate(background):
        z = np.asarray(z, dtype=np.float64)
        log_base = float(z @ beta)
        for patient, x in enumerate(X):
            a = np.expm1((x - z) * beta)
            integral = np.zeros_like(beta)
            for q in range(0, m_q, node_block):
                factors = nodes[q:q + node_block, None] * a[None, :]
                np.log1p(factors, out=factors)
                log_product = log_base + factors.sum(axis=1)
                np.subtract(log_product[:, None], factors, out=factors)
                np.exp(factors, out=factors)
                factors *= weights[q:q + node_block, None]
                integral += factors.sum(axis=0)
            phi[patient] += a * integral / len(background)
        if progress is not None:
            progress(k + 1, len(background))
    return phi
