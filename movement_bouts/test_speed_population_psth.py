"""Speed boundaries and equal-neuron averaging in the population PSTH notebook."""

import contextlib
import io
import json
from pathlib import Path
import pickle
import unittest
from unittest.mock import mock_open, patch

import numpy as np
import pandas as pd


NOTEBOOK = json.loads(Path(__file__).with_name('early_speed_population_psth.ipynb').read_text(encoding='utf-8'))
SOURCES = [''.join(c['source']) for c in NOTEBOOK['cells'] if c['cell_type'] == 'code']


class SpeedPopulationPsthTests(unittest.TestCase):
    def load(self, bouts, sessions):
        scope = {}
        exec(SOURCES[0], scope)
        scope['display'] = lambda *args: None
        with patch('pandas.read_pickle', side_effect=[bouts, sessions]), patch.object(
            Path, 'read_text', return_value=json.dumps({'DT': .1, 'EARLY_SPEED_WINDOW_S': 1.})
        ), contextlib.redirect_stdout(io.StringIO()):
            exec(SOURCES[1], scope)
        return scope

    def test_boundaries_are_half_open_and_missing_or_negative_speeds_are_excluded(self):
        speed = [0, .499, .5, .999, 1, 1.499, 1.5, 1.999, 2, 2.999,
                 3, 3.999, 4, 5.999, 6, 9.999, 10, 30, np.nan, -1, np.inf]
        bouts = pd.DataFrame({'idx': 0, 'condition': 0, 'early_mean_speed': speed}, index=np.arange(101, 122))
        sessions = pd.DataFrame({'feature_file': ['sample_feature_ss']})
        scope = self.load(bouts, sessions)
        self.assertEqual(scope['bouts'].index.tolist(), list(range(101, 119)))
        self.assertEqual(scope['bouts']['speed_group'].tolist(), np.repeat(np.arange(9), 2).tolist())

    def test_population_weights_neurons_equally_and_reports_neural_bounds(self):
        bouts = pd.DataFrame({
            'idx': [0, 0, 0, 0, 1, 1], 'condition': [0, 0, 1, 0, 0, 1],
            'onset_bin': [3, 10, 16, 0, 3, 16],
            'early_mean_speed': [.5, .5, 2., .1, .5, 2.],
        }, index=[17, 4, 93, 28, 117, 193])
        sessions = pd.DataFrame({'feature_file': ['sample_feature_ss'] * 2,
                                 'session': [0, 1], 'n_bins': [20, 20]})
        scope = self.load(bouts, sessions)
        scope.update(WINDOW_S=(-.2, .4), DISPLAY_WINDOW_S=(-.2, .4),
                     BASELINE_WINDOW_S=(-.2, 0), BIN_WIDTH_S=.1, SMOOTHING_SIGMA_S=0)
        a, b = np.zeros((2, 20)), np.zeros((1, 20))
        a[:, 3] = [1, 3]
        a[:, 10] = [1, 3]
        a[:, 16] = [2, 4]
        b[:, 3] = 9
        b[:, 16] = 5
        with patch.object(Path, 'open', mock_open()), patch.object(
            pickle, 'load', side_effect=[[a, b], [['a0', 'a1'], ['b0']]]
        ), contextlib.redirect_stdout(io.StringIO()):
            exec(SOURCES[2], scope)
        tracking = scope['speed_population_psths'][(0, 1)]
        playback = scope['speed_population_psths'][(1, 4)]
        np.testing.assert_allclose(tracking.iloc[:, 2], [10, 30, 90])
        self.assertAlmostEqual(tracking.iloc[:, 2].mean(), 130 / 3)
        np.testing.assert_allclose(playback.iloc[:, 2], [20, 40, 50])
        self.assertEqual(tracking.index.names, ['idx', 'neuron_id'])
        counts = scope['population_counts']
        self.assertEqual(counts.loc[(0, 1), 'n_bouts'], 3)
        self.assertEqual(counts.loc[(0, 1), 'n_units'], 3)
        self.assertEqual(counts.loc[(0, 1), 'n_sessions'], 2)
        self.assertEqual(counts.loc[(0, 0), 'n_excluded'], 1)
        self.assertTrue(scope['speed_population_psths'][(0, 0)].empty)
        with contextlib.redirect_stdout(io.StringIO()), patch('matplotlib.pyplot.show'):
            exec(SOURCES[3], scope)
        scope['plt'].close(scope['fig'])
        tracking_bins = scope['binned_neuron_psths'][(0, 1)]
        np.testing.assert_allclose(tracking_bins.iloc[:, 2], [10, 30, 90])
        self.assertTrue(tracking_bins.index.equals(tracking.index))
        with contextlib.redirect_stdout(io.StringIO()), patch('matplotlib.pyplot.show'):
            exec(SOURCES[4], scope)
            exec(SOURCES[5], scope)
        self.assertAlmostEqual(scope['bin_means'][0][1, 2], 130 / 3)
        expected_sem = np.std([10, 30, 90], ddof=1) / np.sqrt(3)
        self.assertAlmostEqual(scope['bin_sems'][0][1, 2], expected_sem)
        self.assertIs(scope['axes_heatmap'][0].images[0].norm, scope['axes_heatmap'][1].images[0].norm)
        self.assertEqual(scope['bin_starts'][2], 0)
        self.assertTrue(np.isnan(scope['bin_means'][0][0]).all())
        for interval in [0, len(scope['bin_starts']) - 1]:
            fig, axes = scope['plot_speed_interval'](interval)
            self.assertTrue(axes[0].get_shared_y_axes().joined(axes[0], axes[1]))
            scope['plt'].close(fig)
        with contextlib.redirect_stdout(io.StringIO()), patch('pickle.load') as disk_read:
            exec(SOURCES[6], scope)
            scope['interval_slider'].value = len(scope['bin_starts']) - 1
            disk_read.assert_not_called()
        scope['interval_slider'].close()
        scope['interval_output'].close()
        scope['plt'].close('all')

    def test_100ms_quantification_uses_unsmoothed_counts(self):
        bouts = pd.DataFrame({'idx': [0], 'condition': [0], 'onset_bin': [30],
                               'early_mean_speed': [.6]}, index=[17])
        sessions = pd.DataFrame({'feature_file': ['sample_feature_ss'], 'session': [0], 'n_bins': [100]})
        scope = self.load(bouts, sessions)
        scope.update(DT=.01, WINDOW_S=(-.2, .4), DISPLAY_WINDOW_S=(-.2, .4),
                     BASELINE_WINDOW_S=(-.2, 0), BIN_WIDTH_S=.01, SMOOTHING_SIGMA_S=.03)
        recording = np.zeros((1, 100))
        recording[0, 30] = 1
        with patch.object(Path, 'open', mock_open()), patch.object(
            pickle, 'load', side_effect=[[recording], [['unit0']]]
        ), contextlib.redirect_stdout(io.StringIO()):
            exec(SOURCES[2], scope)
        # One spike in the first 100 ms gives exactly 10 spikes/s. Symmetric
        # smoothing would move part of this impulse into the preceding interval.
        bins = scope['binned_neuron_psths'][(0, 1)]
        self.assertEqual(bins.iloc[0, 2], 10.)
        self.assertEqual(bins.iloc[0, 1], 0.)
        native = scope['speed_population_psths'][(0, 1)]
        visible = (native.columns >= 0) & (native.columns < .1)
        self.assertLess(native.loc[:, visible].mean(axis=1).iloc[0], 9.)


if __name__ == '__main__':
    unittest.main()
