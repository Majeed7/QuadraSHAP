"""Standalone Experiment 3 checks; run with python, without the C++ test suite."""
import sys
import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import exp3_synthetic_recovery as exp


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.cfg = exp.configuration("quick", 1, 60)
        r = np.random.default_rng(24)
        self.model = SimpleNamespace(X_fit_=r.normal(size=(12, 6)),
                        dual_coef_=r.normal(size=12), gamma=.12, intercept_=2.1)
        self.x = r.normal(size=6)
        self.bg = r.normal(size=(3, 6))

    def brute(self):
        import math
        d = len(self.x)
        masks = np.array([[(b >> j) & 1 for j in range(d)] for b in range(2**d)])
        values = exp.MaskGame(self.model, self.x, self.bg)(masks)
        phi = np.zeros(d)
        for i in range(d):
            for b, m in enumerate(masks):
                if not m[i]:
                    k = int(m.sum())
                    phi[i] += (values[b | (1 << i)] - values[b]) / (d * math.comb(d-1, k))
        return phi

    def test_game_matches_masked_model_and_counts(self):
        masks = np.random.default_rng(1).integers(0, 2, (15, 6))
        game = exp.MaskGame(self.model, self.x, self.bg)
        actual = game(masks)
        direct = [np.mean(rbf_kernel(np.where(m, self.x, self.bg), self.model.X_fit_,
                         gamma=self.model.gamma) @ self.model.dual_coef_ + self.model.intercept_) for m in masks]
        np.testing.assert_allclose(actual, direct, atol=2e-14)
        self.assertEqual(game.calls, 15)

    def test_independent_reference_and_package_against_brute_force(self):
        ref = self.brute()
        np.testing.assert_allclose(exp.independent_reference(self.model, self.x, self.bg, 3), ref, atol=2e-14)
        phi, _ = exp.quad(self.model, self.x, self.bg, exact=True)
        np.testing.assert_allclose(phi, ref, atol=2e-14)
        phi, info = exp.quad(self.model, self.x, self.bg, eps=1e-6)
        self.assertLessEqual(np.max(np.abs(phi-ref)), 1e-6)
        self.assertLessEqual(info["bound"], 1e-6)

    def test_kernelshap_full_enumeration(self):
        phi, meta = exp.sampler(self.model, self.x, self.bg, "kernel", 1000, 31)
        np.testing.assert_allclose(phi, self.brute(), atol=1e-12)
        self.assertIn("l1_reg=0", meta["implementation"])

    def test_sampler_reproducibility(self):
        p, _ = exp.sampler(self.model, self.x, self.bg, "permutation", 2, 11)
        q, _ = exp.sampler(self.model, self.x, self.bg, "permutation", 2, 11)
        np.testing.assert_array_equal(p, q)
        p, _ = exp.sampler(self.model, self.x, self.bg, "kernel", 20, 11)
        q, _ = exp.sampler(self.model, self.x, self.bg, "kernel", 20, 11)
        np.testing.assert_array_equal(p, q)

    def test_data_independence_and_equal_feature_scale(self):
        cfg = dict(self.cfg, n_train=10000)
        a = exp.make_data(cfg, 9)
        b = exp.make_data(dict(cfg, n_test=200), 9)
        np.testing.assert_array_equal(a["X_train"], b["X_train"])
        np.testing.assert_array_equal(a["y_train"], b["y_train"])
        self.assertEqual(a["informative"].sum(), 6)
        variance = a["X_train"].var(axis=0)
        self.assertTrue(np.all((variance > .9) & (variance < 1.1)))

    def test_ties_and_recovery(self):
        mask = np.array([1, 1, 0, 0, 0], bool)
        tied = exp.recovery_metrics(np.zeros(5), mask)
        self.assertAlmostEqual(tied["average_precision"], .4)
        self.assertAlmostEqual(tied["auroc"], .5)
        perfect = exp.recovery_metrics(np.array([4., 3., 2., 1., 0.]), mask)
        self.assertEqual(perfect["average_precision"], 1)
        self.assertEqual(perfect["precision_at_s"], 1)

    def test_job_names_do_not_collide_on_file_suffixes(self):
        names = [exp.job_name(j) for j in exp.jobs(self.cfg, 0)]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(Path(n).suffix == "" for n in names))

    def test_full_profile_has_thirty_inputs_and_retains_sampler_repeats(self):
        cfg = exp.configuration("full", 4, 180)
        self.assertEqual(cfg["seeds"], [0, 1, 2, 3, 4])
        self.assertEqual(cfg["n_instances"], 6)
        self.assertEqual(cfg["sampler_repeats"], 3)
        planned = [job for seed in cfg["seeds"] for job in exp.jobs(cfg, seed)]
        self.assertEqual(len({(j["seed"], j["instance"]) for j in planned}), 30)
        self.assertEqual(len(planned), 720)

    def test_incomplete_sampler_repeats_are_not_plotted(self):
        import exp3_recovery_analysis as analysis
        cfg = dict(self.cfg, sampler_repeats=2)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            exp.atomic_json(output / "config.json", cfg)
            exp.atomic_json(output / "provenance.json", {"source_hashes": {"experiments/exp3_synthetic_recovery.py": "test"}})
            fit = output / "seed_97/fit.npz"
            exp.atomic_npz(fit, informative=np.arange(20) < 6)
            exp.atomic_json(fit.with_suffix(".json"), dict(seed=97, sha256=exp.digest(fit),
                            validation_r2=.9, test_r2=.8, gamma=.005, alpha=.01))
            budget = cfg["kernel_multipliers"][0]*cfg["d"]+128
            for job in exp.jobs(cfg, 97):
                if job["method"] == "reference" or (job["method"] == "kernel" and job["budget"] == budget and job["repeat"] == 0):
                    stem = output / "seed_97/jobs" / exp.job_name(job)
                    exp.atomic_npz(stem.with_suffix(".npz"), phi=np.arange(20, dtype=float))
                    exp.atomic_json(stem.with_suffix(".json"), dict(job, status="ok", seconds=1.,
                              fit_sha256=exp.digest(fit), result_sha256=exp.digest(stem.with_suffix(".npz")),
                              max_abs_error=.01, relative_l2_error=.01))
            with patch.object(analysis, "plot"):
                result = analysis.analyze(output)
            import csv
            with (output / "summary.csv").open() as fh:
                rows = list(csv.DictReader(fh))
            kernel = next(r for r in rows if r["method"] == "kernel" and float(r["budget"]) == budget)
            self.assertEqual(kernel["n_fits"], "0")
            self.assertEqual(kernel["complete"], "False")
            self.assertFalse(result["collection_complete"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
