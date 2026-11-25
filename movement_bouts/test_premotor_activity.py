"""Checks for extraction windows and optional bout/neuron baseline subtraction."""

import unittest

import numpy as np
import pandas as pd

from evaluation import validate_data_labels
from utils import extract_premotor_bouts, prepare_pca_input, fit_pca_by_condition, project_onto_tracking_pca


class PremotorActivityTests(unittest.TestCase):
    def setUp(self):
        self.dt = 0.005
        # Different baselines for every bout/neuron; explicit nonzero values in
        # the unused first 200 ms catch errors in absolute window positioning.
        self.arrays = []
        for baseline, changes in (([2, 4], [1, -1]), ([6, 0], [-2, 3])):
            counts = np.full((2, 240), 99.0)
            counts[:, 40:140] = np.asarray(baseline)[:, None]
            for t in range(5):
                counts[:, 140 + t * 20:160 + t * 20] = (
                    np.asarray(baseline) + np.asarray(changes) * (t + 1) / 5
                )[:, None]
            self.arrays.append(counts)
        self.neuronal = pd.DataFrame({"spike_counts": self.arrays}, index=[51, 9])
        # Non-sorted IDs and another session exercise original reference ordering.
        self.bouts = pd.DataFrame({"idx": [7, 99, 7], "condition": [0, 0, 1], "initial_position": [3, 5, 12]}, index=[9, 80, 51])
        self.neurons = pd.DataFrame({"neuron_ids": [["neuron_b", "neuron_a"]]}, index=[7])

    def prepare(self, **overrides):
        kwargs = dict(idx=7, dt=self.dt, n_bins=5, extraction_window_s=(-1.2, 0), analysis_window_s=(-0.5, 0))
        kwargs.update(overrides)
        return prepare_pca_input(self.neuronal, self.bouts, self.neurons, **kwargs)

    def test_raw_matches_legacy_averaging_and_preserves_ids_and_neurons(self):
        data, labels = self.prepare()
        expected_arrays = np.stack([self.arrays[1][:, 140:], self.arrays[0][:, 140:]])
        expected = expected_arrays.reshape(2, 2, 5, 20).sum(axis=-1) / 0.1
        np.testing.assert_array_equal(data, expected.transpose(0, 2, 1).reshape(10, 2))
        self.assertEqual(data.columns.tolist(), ["neuron_b", "neuron_a"])
        self.assertEqual(data.index.get_level_values("bout_id").tolist(), [9] * 5 + [51] * 5)
        np.testing.assert_allclose(data.index.get_level_values("bin_start_s"), [-0.5, -0.4, -0.3, -0.2, -0.1] * 2)
        np.testing.assert_allclose(labels.bin_end_s, [-0.4, -0.3, -0.2, -0.1, 0] * 2)
        validate_data_labels(data, labels)

    def test_subtraction_uses_each_bout_and_neuron_baseline_without_mutation(self):
        original = [a.copy() for a in self.arrays]
        raw, raw_labels = self.prepare()
        corrected, labels = self.prepare(activity_mode="baseline_subtracted", baseline_window_s=(-1, -0.5))
        expected_baseline = np.repeat(np.array([[1200, 0], [400, 800]]), 5, axis=0)
        np.testing.assert_allclose(raw - corrected, expected_baseline)
        self.assertTrue((corrected.to_numpy() < 0).any())
        pd.testing.assert_frame_equal(labels, raw_labels)
        for before, after in zip(original, self.arrays):
            np.testing.assert_array_equal(before, after)
        validate_data_labels(corrected, labels)

    def test_other_bin_counts_and_analysis_intervals_use_correct_time_axis(self):
        for n_bins in (1, 2, 5):
            with self.subTest(n_bins=n_bins):
                raw, _ = self.prepare(n_bins=n_bins)
                corrected, _ = self.prepare(n_bins=n_bins, activity_mode="baseline_subtracted", baseline_window_s=(-1, -0.5))
                np.testing.assert_allclose(raw - corrected, np.repeat([[1200, 0], [400, 800]], n_bins, axis=0))
        data, labels = self.prepare(n_bins=2, analysis_window_s=(-0.4, -0.2))
        np.testing.assert_allclose(data.index.get_level_values("bin_start_s"), [-0.4, -0.3] * 2)
        np.testing.assert_allclose(labels.bin_end_s, [-0.3, -0.2] * 2)
        # Raw mode does not require a baseline, including with legacy 500 ms data.
        legacy = self.neuronal.copy()
        legacy["spike_counts"] = [a[:, 140:] for a in self.arrays]
        legacy_data, _ = prepare_pca_input(legacy, self.bouts, self.neurons, idx=7, dt=self.dt, n_bins=5,
                                         extraction_window_s=(-0.5, 0), analysis_window_s=(-0.5, 0))
        current, _ = self.prepare()
        pd.testing.assert_frame_equal(current, legacy_data)

    def test_changing_baseline_interval_uses_its_own_counts_and_duration(self):
        raw, _ = self.prepare()
        corrected, _ = self.prepare(activity_mode="baseline_subtracted", baseline_window_s=(-1.2, -1.0))
        # The first 200 ms contain 99 counts/bin, rather than the later baseline.
        expected_rate = 99 * 40 / 0.2
        np.testing.assert_allclose(raw - corrected, expected_rate)

    def test_invalid_windows_metadata_and_neuron_axis_are_rejected(self):
        for kwargs, message in (
            ({"analysis_window_s": (-1.3, 0)}, "contained"),
            ({"analysis_window_s": (-0.501, 0)}, "align"),
            ({"activity_mode": "divide"}, "activity_mode"),
            ({"activity_mode": "baseline_subtracted"}, "two finite"),
            ({"activity_mode": "baseline_subtracted", "baseline_window_s": (-0.6, -0.4)}, "end at or before"),
            ({"activity_mode": "baseline_subtracted", "baseline_window_s": (-1.3, -0.5)}, "contained"),
            ({"extraction_window_s": (-1.1, 0)}, "array length"),
            ({"n_bins": 3}, "divide"),
            ({"dt": 0}, "positive finite"),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                self.prepare(**kwargs)
        self.neurons.at[7, "neuron_ids"] = ["one"]
        with self.assertRaisesRegex(ValueError, "neuron IDs"):
            self.prepare()

    def test_extraction_is_half_open_skips_bounds_and_copies_original_counts(self):
        neural = np.arange(800).reshape(2, 400)
        bouts = pd.DataFrame({"onset_bin": [239, 240, 399, 400]}, index=[11, 22, 33, 44])
        extracted, excluded = extract_premotor_bouts(neural, bouts, dt=self.dt, extraction_window_s=(-1.2, 0))
        self.assertEqual(extracted.index.tolist(), [22, 33])
        self.assertEqual(excluded.index.tolist(), [11, 44])
        np.testing.assert_array_equal(extracted.at[22, "spike_counts"], neural[:, :240])
        np.testing.assert_array_equal(extracted.at[33, "spike_counts"], neural[:, 159:399])
        self.assertFalse(np.shares_memory(extracted.at[22, "spike_counts"], neural))
        shifted, _ = extract_premotor_bouts(neural, bouts.loc[[22]], dt=self.dt, extraction_window_s=(-1, -0.5))
        np.testing.assert_array_equal(shifted.at[22, "spike_counts"], neural[:, 40:140])

    def test_tracking_projection_accepts_corrected_inputs_and_keeps_labels(self):
        data, labels = self.prepare(activity_mode="baseline_subtracted", baseline_window_s=(-1, -0.5))
        fits = fit_pca_by_condition(data, labels)
        projected = project_onto_tracking_pca(data, labels, fits[0]["model"])
        for condition in (0, 1):
            rows = projected[condition]["results"]
            selected = data.loc[labels.condition == condition]
            self.assertTrue(rows.index.equals(selected.index))
            np.testing.assert_allclose(rows[["PC1", "PC2"]], fits[0]["model"].transform(selected))


if __name__ == "__main__":
    unittest.main()
