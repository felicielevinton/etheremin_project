"""Checks for matched observations and independent unsupervised UMAP fits."""

import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from umap_analysis import fit_umap_by_condition, plot_umap_comparison, make_interactive_umap_comparison
import evaluation


class UmapTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.MultiIndex.from_product([[11, 25, 38, 49], [-.5, -.4]],
                                                names=["bout_id", "bin_start_s"])
        self.data = pd.DataFrame(np.arange(24).reshape(8, 3), index=self.index,
                                 columns=["neuron_c", "neuron_a", "neuron_b"])
        self.labels = pd.DataFrame({"condition": [0]*4 + [1]*4,
                                    "initial_position": [np.nan, 2, 4, 6, 8, 10, 12, 14],
                                    "initial_direction": [-1, 1, 0, np.nan]*2}, index=self.index)

    def test_condition_fits_preserve_ids_and_neuron_order_without_targets(self):
        # Capture the actual fit inputs: behavior must never be passed to UMAP.
        with patch("umap.UMAP") as factory:
            factory.return_value.fit_transform.side_effect = [np.ones((4, 3)), np.zeros((4, 3))]
            fits, summary = fit_umap_by_condition(self.data, self.labels, n_neighbors=3)
        self.assertEqual(factory.call_count, 2)
        self.assertEqual(factory.call_args.kwargs["n_components"], 3)
        self.assertTrue(summary["status"].eq("ok").all())
        self.assertEqual(summary["n_bouts"].tolist(), [2, 2])
        for condition in (0, 1):
            expected = self.data.loc[self.labels["condition"].eq(condition)]
            actual = factory.return_value.fit_transform.call_args_list[condition]
            pd.testing.assert_frame_equal(actual.args[0], expected)
            self.assertEqual(len(actual.args), 1)
            self.assertEqual(actual.kwargs, {})
            self.assertTrue(fits[condition]["results"].index.equals(expected.index))
            self.assertEqual(fits[condition]["neuron_ids"], self.data.columns.tolist())

    def test_small_conditions_skip_and_misalignment_rejected(self):
        fits, summary = fit_umap_by_condition(self.data, self.labels, n_neighbors=4)
        self.assertEqual(fits, {})
        self.assertTrue(summary["status"].eq("skipped").all())
        with self.assertRaises(ValueError):
            fit_umap_by_condition(self.data, self.labels.iloc[::-1])
        with self.assertRaises(ValueError):
            fit_umap_by_condition(self.data, self.labels, n_neighbors=1)

    def test_shared_evaluations_record_umap_provenance_and_keep_prediction_ids(self):
        rows = self.labels.assign(UMAP1=np.arange(8), UMAP2=np.sin(np.arange(8)),
                                  UMAP3=np.cos(np.arange(8)), bin_number=[0, 1]*4,
                                  bin_end_s=[-.4, -.3]*4)
        cache = {"representation": "UMAP", "projection_basis": "separate_condition_umaps",
                 "settings": {"random_state": None, "activity_mode": "raw"},
                 "session_info": pd.DataFrame({"animal": ["test"], "headstage": [0], "session": [0]}, index=[7]),
                 "sessions": {7: {c: {"results": rows.loc[rows["condition"].eq(c)]} for c in (0, 1)}}}
        columns = ["UMAP1", "UMAP2", "UMAP3"]
        regression = evaluation.run_label_regressions(cache, target="initial_position", columns=columns)
        logistic = evaluation.run_direction_logistics(cache, columns=columns)
        for result in (regression, logistic):
            self.assertEqual(result["settings"]["representation"], "UMAP")
            self.assertEqual(result["settings"]["representation_settings"], cache["settings"])
            self.assertNotIn("pca_settings", result["settings"])
            for key, fit in result["fits"].items():
                self.assertTrue(fit["predictions"].index.isin(cache["sessions"][7][key[1]]["results"].index).all())
        self.assertEqual(regression["summary"].loc[(7, 0), "n_excluded_target"], 1)
        self.assertEqual(logistic["summary"]["n_excluded_target"].tolist(), [2, 2])
        fig, axes = evaluation.plot_regression_summary(logistic["summary"], score="balanced_accuracy", representation="UMAP")
        self.assertTrue(all("per UMAP" in ax.get_title() for ax in axes))
        plt.close(fig)

    def test_plot_keeps_unknown_directions_and_missing_position(self):
        results = self.labels.assign(UMAP1=np.arange(8), UMAP2=np.arange(8), UMAP3=np.arange(8))
        fits = {c: {"results": results.loc[results["condition"].eq(c)]} for c in (0, 1)}
        bundle = {"fits": {"raw": fits, "baseline_subtracted": fits}, "position_limits": (0, 28),
                  "settings": {"idx": 0, "n_neighbors": 3, "random_state": 42}}
        for color_by in ("position", "direction"):
            with self.subTest(color_by=color_by):
                fig, axes = plot_umap_comparison(bundle, color_by=color_by)
                for ax in axes.flat:
                    self.assertEqual(ax.name, "3d")
                    self.assertEqual(sum(len(c.get_offsets()) for c in ax.collections), 4)
                plt.close(fig)
                interactive = make_interactive_umap_comparison(bundle, color_by=color_by)
                self.assertEqual(sum(len(t.x) for t in interactive.data), 16)
                self.assertEqual({t.marker.symbol for t in interactive.data}, {"circle"})
                for trace in interactive.data:
                    self.assertEqual(len(trace.x), len(trace.z))
                self.assertIsInstance(interactive.to_plotly_json()["data"][0]["x"], list)
                if color_by == "direction":
                    self.assertEqual({t.name: t.marker.color for t in interactive.data},
                                     {"Leftward": "#0000ff", "Rightward": "#ff0000", "Unknown direction": "#8c8c8c"})
                    self.assertFalse(any(t.marker.coloraxis for t in interactive.data))
                else:
                    gray = next(t for t in interactive.data if t.name == "Unknown position")
                    self.assertEqual(gray.marker.color, "#8c8c8c")
                    self.assertEqual(len(gray.x), 1)



if __name__ == "__main__":
    unittest.main()
