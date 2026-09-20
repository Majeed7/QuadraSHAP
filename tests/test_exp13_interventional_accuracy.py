"""The matched-background reference preserves the full interventional game."""
import numpy as np
from scipy import sparse

from experiments.exp13_interventional_accuracy import reference, tables_for_background
from quadrashap.product_games.budget import GameSummary
from tests.naive_shapley import naive_shapley_absent


def test_compressed_reference_matches_exhaustive_full_game():
    z = np.array([[0.0, 0.2, 0.0, 0.3, 0.0, 0.1],
                  [0.4, 0.0, 0.1, 0.0, 0.2, 0.0],
                  [0.0, 0.1, 0.5, 0.0, 0.0, 0.3]])
    x = np.array([0.2, 0.0, 0.4, 0.0, 0.0, 0.1])
    bg = np.array([[0.1, 0.0, 0.0, 0.3, 0.0, 0.1],
                   [0.2, 0.0, 0.8, 0.0, 0.0, 0.0]])
    alpha = np.array([0.7, -0.2, 0.5])
    gamma = 0.8
    phi, _, diag = reference(sparse.csr_matrix(z), alpha, gamma, x, bg, eps=1e-12)
    brute = np.zeros(len(x))
    for b in bg:
        for a, zr in zip(alpha, z):
            present = np.exp(-gamma * np.square(x - zr))
            absent = np.exp(-gamma * np.square(b - zr))
            brute += (a / len(bg)) * naive_shapley_absent(present, absent)
    np.testing.assert_allclose(phi, brute, rtol=0, atol=2e-14)
    expected_delta = sum(a * (np.exp(-gamma * np.square(x - zr).sum()) -
                              np.mean([np.exp(-gamma * np.square(b - zr).sum()) for b in bg]))
                         for a, zr in zip(alpha, z))
    np.testing.assert_allclose(phi.sum(), expected_delta, rtol=0, atol=2e-14)
    assert diag["certified_bound"] <= 1e-12


def test_reduced_certificate_amplitude_equals_full_game():
    z = sparse.csr_matrix([[0.1, 0.0, 0.2, 0.0], [0.0, 0.4, 0.0, 0.2]])
    x = np.array([0.2, 0.0, 0.3, 0.0])
    b = np.array([0.1, 0.0, 0.3, 0.5])
    alpha = np.array([0.6, -0.5])
    gamma = 1.2
    z2 = z.toarray()
    full_u = np.exp(-gamma * np.square(z2 - x).astype(float))
    full_ut = np.exp(-gamma * np.square(z2 - b).astype(float))
    full = GameSummary(d=4).update(full_u - full_ut, full_ut, alpha)
    dist2_x = np.square(z2 - x).sum(axis=1)
    varying, K, Ut, weight = tables_for_background(z, alpha, gamma, x, b, dist2_x, 1)
    reduced = GameSummary(d=len(varying)).update(K, Ut, weight)
    mapped = np.zeros(4)
    mapped[varying] = reduced.A
    np.testing.assert_allclose(mapped, full.A, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(reduced.lambda_max, full.lambda_max, rtol=1e-13)
