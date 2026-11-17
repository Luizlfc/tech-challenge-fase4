"""Testes das estatísticas de drift (PSI e KS)."""
import numpy as np

from src.metrics import ks_test, psi


def test_psi_same_distribution_is_small():
    rng = np.random.default_rng(1)
    assert psi(rng.normal(0, 1, 5000), rng.normal(0, 1, 5000)) < 0.02


def test_psi_shifted_distribution_is_large():
    rng = np.random.default_rng(2)
    assert psi(rng.normal(0, 1, 5000), rng.normal(1, 1, 5000)) > 0.20


def test_psi_categorical():
    ref = np.array([0] * 800 + [1] * 200)
    cur = np.array([0] * 400 + [1] * 600)
    assert psi(ref, cur) > 0.20


def test_ks_detects_shift():
    rng = np.random.default_rng(3)
    _, p_same = ks_test(rng.normal(0, 1, 2000), rng.normal(0, 1, 2000))
    _, p_diff = ks_test(rng.normal(0, 1, 2000), rng.normal(0.5, 1, 2000))
    assert p_same > 0.01 and p_diff < 1e-6
