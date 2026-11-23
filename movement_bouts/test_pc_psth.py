"""Check that PC PSTHs are fixed-basis projections of matched neural bouts."""

import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from pc_psth import project_pc_psths, plot_pc_psths, summarize_pc_psths
from psth import extract_neuron_counts, prepare_psth_rates


class PcPsthTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.ids = ["unit_a", "unit_b", "unit_c", "unit_d"]
        self.neural = rng.poisson(.5, size=(4, 600)).astype(float)
        self.model = PCA(n_components=3).fit(pd.DataFrame(rng.normal(size=(50, 4)) + 5, columns=self.ids))
        self.bouts = pd.DataFrame({"onset_bin": [5, 100, 200, 300, 595], "condition": [0, 0, 1, 0, 1],
                                   "initial_direction": [0, -1, 1, -1, 0]},
                                  index=[12, 20, 33, 48, 50])

    def compute(self, **kwargs):
        return project_pc_psths(self.neural, self.ids, self.bouts,
                                 {"raw": self.model, "baseline_subtracted": self.model},
                                 dt=.01, window_s=(-.6, .6), bin_width_s=.02,
                                 baseline_window_s=(-.5, -.3), smoothing_sigma_s=0, **kwargs)

    def test_scores_match_sklearn_transform_for_both_neural_preprocessings(self):
        result = self.compute()
        self.assertEqual(result["labels"].index.tolist(), [20, 33, 48])
        self.assertEqual(result["excluded_bouts"].index.tolist(), [12, 50])
        for mode in ("raw", "baseline_subtracted"):
            neural_rates = []
            for neuron in range(4):
                counts, labels, _ = extract_neuron_counts(self.neural, self.bouts, neuron=neuron,
                                                          dt=.01, window_s=(-.6, .6))
                rates, _ = prepare_psth_rates(counts, dt=.01, window_s=(-.6, .6), bin_width_s=.02,
                                              baseline_window_s=(-.5, -.3) if mode == "baseline_subtracted" else None)
                neural_rates.append(rates)
            rates = np.stack(neural_rates, axis=-1)
            expected = self.model.transform(pd.DataFrame(rates.reshape(-1, 4), columns=self.ids)).reshape(3, 60, 3)
            for condition in (0, 1):
                actual = result["modes"][mode]["conditions"][condition]
                selected = labels["condition"].eq(condition).to_numpy()
                np.testing.assert_allclose(actual["bout_scores"], expected[selected], atol=1e-12)
                np.testing.assert_allclose(actual["mean"], expected[selected].mean(axis=0), atol=1e-12)
                if selected.sum() > 1:
                    np.testing.assert_allclose(actual["sem"], expected[selected].std(axis=0, ddof=1) / np.sqrt(selected.sum()), atol=1e-12)
                else:
                    self.assertTrue(np.isnan(actual["sem"]).all())

    def test_baseline_subtraction_preserves_saved_pca_center(self):
        result = self.compute()
        visible = (result["centers_s"] >= -.5) & (result["centers_s"] < -.3)
        zero_neural_score = -self.model.mean_ @ self.model.components_.T
        actual = result["modes"]["baseline_subtracted"]["conditions"][0]["bout_scores"][:, visible].mean(axis=1)
        np.testing.assert_allclose(actual, np.tile(zero_neural_score, (2, 1)), atol=1e-12)

    def test_identity_and_model_checks(self):
        with self.assertRaises(ValueError):
            project_pc_psths(self.neural, self.ids[::-1], self.bouts,
                              {"raw": self.model, "baseline_subtracted": self.model}, dt=.01)
        self.model.whiten = True
        with self.assertRaises(ValueError):
            self.compute()

    def test_missing_condition_and_plot(self):
        self.bouts["condition"] = 0
        result = self.compute()
        self.assertEqual(result["modes"]["raw"]["conditions"][1]["n_bouts"], 0)
        fig, axes = plot_pc_psths(result, display_window_s=(-.4, .4))
        self.assertEqual(axes.shape, (2, 3))
        self.assertEqual(axes[0, 0].lines[0].get_color(), "red")
        self.assertEqual(axes[0, 0].lines[1].get_color(), "black")
        plt.close(fig)

    def test_cached_summaries_match_direct_direction_projections(self):
        full = self.compute()
        summaries = summarize_pc_psths(full)
        original = self.bouts
        for direction in (None, -1, 1):
            self.bouts = original if direction is None else original.loc[original.initial_direction.eq(direction)]
            direct = self.compute()
            cached = summaries[direction]
            self.assertEqual(cached["n_complete_bouts"], len(direct["labels"]))
            self.assertEqual(cached["n_excluded_bouts"], len(direct["excluded_bouts"]))
            for mode in ("raw", "baseline_subtracted"):
                self.assertIs(cached["modes"][mode]["model"], full["modes"][mode]["model"])
                for condition in (0, 1):
                    observed = cached["modes"][mode]["conditions"][condition]
                    expected = direct["modes"][mode]["conditions"][condition]
                    self.assertNotIn("bout_scores", observed)
                    self.assertEqual(observed["bout_ids"], expected["bout_ids"])
                    np.testing.assert_allclose(observed["mean"], expected["mean"], equal_nan=True)
                    np.testing.assert_allclose(observed["sem"], expected["sem"], equal_nan=True)

    def test_unknown_directions_only_enter_all_bouts_summary(self):
        self.bouts.loc[20, "initial_direction"] = 0
        summaries = summarize_pc_psths(self.compute())
        self.assertEqual(summaries[None]["n_complete_bouts"], 3)
        self.assertEqual(summaries[-1]["n_complete_bouts"], 1)
        self.assertEqual(summaries[1]["n_complete_bouts"], 1)


if __name__ == "__main__":
    unittest.main()
