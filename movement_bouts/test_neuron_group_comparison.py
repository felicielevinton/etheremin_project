"""Check within-session matching and pairwise group-score summaries."""

import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evaluation import pair_neuron_group_scores, plot_neuron_group_comparison, plot_neuron_group_pairwise


class GroupComparisonTests(unittest.TestCase):
    def setUp(self):
        self.summary = pd.DataFrame([
            ['All neurons', 10, 0, .2, 'ok', 10, 20, 4],
            ['All neurons', 11, 0, .9, 'ok', 20, 30, 6],
            ['All neurons', 10, 1, .5, 'ok', 10, 20, 4],
            ['Motor', 10, 0, .3, 'ok', 5, 20, 4],
            ['Motor', 10, 1, .4, 'skipped', 5, 20, 4],
            ['Motor', 99, 0, 1., 'ok', 5, 20, 4],
            ['Sensory', 10, 0, .1, 'ok', 5, 20, 4],
            ['Sensory', 11, 0, np.nan, 'ok', 5, 30, 6],
        ], columns=['group', 'idx', 'condition', 'r2', 'status', 'n_neurons', 'n_observations', 'n_bouts']).set_index(['group', 'idx', 'condition'])

    def test_only_same_session_condition_successful_finite_pairs(self):
        paired = pair_neuron_group_scores(self.summary, score='r2')
        self.assertEqual(paired.index.tolist(), [('Motor', 10, 0), ('Sensory', 10, 0)])
        self.assertAlmostEqual(paired.loc[('Motor', 10, 0), 'difference'], .1)
        self.assertEqual(paired.loc[('Motor', 10, 0), 'n_neurons'], 5)
        self.assertEqual(paired.loc[('Motor', 10, 0), 'reference_n_neurons'], 10)

    def test_different_observation_counts_are_rejected(self):
        self.summary.loc[('Motor', 10, 0), 'n_observations'] = 25
        with self.assertRaisesRegex(ValueError, 'different n_observations'):
            pair_neuron_group_scores(self.summary, score='r2')

    def test_pairwise_difference_uses_pair_intersection_and_is_antisymmetric(self):
        fig, axes, table = plot_neuron_group_pairwise(self.summary, score='r2')
        try:
            self.assertEqual(table.loc[(0, 'Motor', 'Sensory'), 'n_pairs'], 1)
            self.assertAlmostEqual(table.loc[(0, 'Motor', 'Sensory'), 'mean_difference'], .2)
            self.assertAlmostEqual(table.loc[(0, 'Sensory', 'Motor'), 'mean_difference'], -.2)
            self.assertEqual(table.loc[(1, 'Motor', 'Sensory'), 'n_pairs'], 0)
            self.assertTrue(np.isnan(table.loc[(1, 'Motor', 'Sensory'), 'mean_difference']))
        finally:
            plt.close(fig)

    def test_comparison_plot_handles_condition_without_pairs(self):
        fig, axes, pairs = plot_neuron_group_comparison(self.summary, score='r2')
        try:
            self.assertEqual(axes.shape, (2, 2))
            self.assertEqual(len(pairs), 2)
            fig.canvas.draw()
        finally:
            plt.close(fig)

    def test_no_valid_fits_keeps_zero_pair_counts(self):
        summary = self.summary.copy()
        summary['status'] = 'skipped'
        fig, axes, table = plot_neuron_group_pairwise(summary, score='r2')
        try:
            self.assertTrue(table.n_pairs.eq(0).all())
            self.assertTrue(table.mean_difference.isna().all())
        finally:
            plt.close(fig)

    def test_requested_type_palette_is_consistent_between_conditions(self):
        from matplotlib.colors import to_rgba
        extra = []
        for group in ['Sensorimotor', 'None']:
            row = self.summary.loc[[('Motor', 10, 0)]].reset_index()
            row['group'] = group
            extra.append(row.set_index(['group', 'idx', 'condition']))
        summary = pd.concat([self.summary, *extra])
        expected = {'Motor': '#2878B5', 'Sensory': '#E6C62E',
                    'Sensorimotor': '#9BD77B', 'None': '#8B8B8B'}
        fig, axes, pairs = plot_neuron_group_comparison(summary, score='r2')
        try:
            groups = [g for g in summary.index.get_level_values('group').unique() if g != 'All neurons']
            for col in (0, 1):
                for collection, group in zip(axes[0, col].collections, groups):
                    np.testing.assert_allclose(collection.get_facecolors()[0], to_rgba(expected[group], alpha=.85))
        finally:
            plt.close(fig)


if __name__ == '__main__':
    unittest.main()
