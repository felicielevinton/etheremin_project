"""Checks for per-bout baseline correction and temporal smoothing."""

import unittest
import numpy as np
from scipy.ndimage import gaussian_filter1d

from psth import prepare_psth_rates, session_neuron_psths, plot_population_psth_by_type


class PsthTests(unittest.TestCase):
    def setUp(self):
        self.settings = dict(dt=0.005, window_s=(-1.2, 1), bin_width_s=0.005,
                             baseline_window_s=(-1, -0.7))
        self.counts = np.zeros((2, 440))
        # Baseline is 60 original bins, independently for each bout.
        self.counts[:, 40:100] = np.array([2, 4])[:, None]
        self.counts[:, 140:340] = np.array([3, 7])[:, None]

    def test_each_bout_uses_own_baseline(self):
        rates, centers = prepare_psth_rates(self.counts, **self.settings)
        np.testing.assert_allclose(rates[:, 40:100], 0)
        np.testing.assert_allclose(rates[:, 140:340], np.array([200, 600])[:, None] * np.ones((2, 200)))
        self.assertAlmostEqual(centers[240], 0.0025)

    def test_smooth_corrected_bouts_before_statistics(self):
        raw, _ = prepare_psth_rates(self.counts, **self.settings)
        smooth, _ = prepare_psth_rates(self.counts, smoothing_sigma_s=.02, **self.settings)
        expected = gaussian_filter1d(raw, 4, axis=1, mode="nearest", truncate=4)
        np.testing.assert_allclose(smooth, expected)
        self.assertFalse(np.array_equal(smooth, raw))

    def test_baseline_uses_counts_before_rebinning(self):
        settings = {**self.settings, "bin_width_s": .025}
        rates, _ = prepare_psth_rates(self.counts, **settings)
        np.testing.assert_allclose(rates[:, 28:68], np.array([200, 600])[:, None] * np.ones((2, 40)))

    def test_reject_outside_or_unaligned_baseline(self):
        for baseline in [(-1.3, -.7), (-1, -.701), (-.5, .1), (-.7, -1)]:
            with self.subTest(baseline=baseline), self.assertRaises(ValueError):
                prepare_psth_rates(self.counts, **{**self.settings, "baseline_window_s": baseline})

    def test_population_means_neurons_not_pooled_bouts(self):
        import pandas as pd
        settings = dict(dt=.005, window_s=(-.01, .01), bin_width_s=.005,
                        baseline_window_s=(-.01, -.005))
        # One recording has two neurons and two bouts, another has one
        # neuron and one bout. Baselines are zero, post-onset rates are known.
        a = np.zeros((2, 20))
        a[:, [5, 6, 12, 13]] = np.array([1, 3])[:, None]
        labels = pd.DataFrame({'onset_bin': [5, 12], 'condition': [0, 0]}, index=[10, 22])
        ta, _, counts = session_neuron_psths(a, labels, **settings)
        self.assertEqual(counts, {0: 2, 1: 0})
        self.assertEqual(ta[1].shape, (0, 4))
        b = np.zeros((1, 20))
        b[:, [5, 6]] = 8
        tb, _, _ = session_neuron_psths(b, labels.iloc[:1], **settings)
        mean = np.concatenate([ta[0], tb[0]]).mean(axis=0)
        np.testing.assert_allclose(mean, [0, 0, 800, 800])

    def test_type_means_exclude_missing_but_retain_none_category(self):
        import pandas as pd
        import matplotlib.pyplot as plt
        index = pd.MultiIndex.from_tuples(
            [(0, 'a'), (0, 'b'), (1, 'c'), (1, 'd')], names=['idx', 'neuron_id'])
        table = pd.DataFrame(np.repeat([[1], [3], [100], [1000]], 4, axis=1), index=index)
        labels = pd.Series(['Motor', 'Motor', 'None', np.nan], index=index)
        fig, axes = plot_population_psth_by_type(
            {0: table, 1: table.iloc[:0]}, {0: labels, 1: labels.iloc[:0]},
            np.array([-.015, -.005, .005, .015]),
            display_window_s=(-.02, .02), title='test')
        try:
            np.testing.assert_allclose(axes[0, 0].lines[0].get_ydata(), 2)
            np.testing.assert_allclose(axes[1, 1].lines[0].get_ydata(), 100)
            # Motor SEM = 1 for the two unit traces [1, 3].
            vertices = axes[0, 0].collections[0].get_paths()[0].vertices[:, 1]
            self.assertEqual(vertices.min(), 1)
            self.assertEqual(vertices.max(), 3)
            self.assertEqual(axes[0, 1].lines[0].get_label(), 'Tracking: 0 neurons')
        finally:
            plt.close(fig)

    def test_type_plot_rejects_reordered_labels(self):
        import pandas as pd
        table = pd.DataFrame(np.zeros((2, 4)), index=['a', 'b'])
        labels = pd.Series(['Motor', 'Sensory'], index=['b', 'a'])
        with self.assertRaisesRegex(ValueError, 'align'):
            plot_population_psth_by_type(
                {0: table, 1: table}, {0: labels, 1: labels},
                np.arange(4), display_window_s=(0, 4), title='test')


if __name__ == "__main__":
    unittest.main()
