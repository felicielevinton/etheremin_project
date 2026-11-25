"""Checks for descriptive label regression; run with unittest discovery."""

import io
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation import (
    fit_label_regression, plot_label_regression,
    run_label_regressions, plot_regression_summary, plot_paired_regressions,
    summarize_starting_position_bias, plot_position_bias_scores, plot_paired_bias_differences,
    fit_direction_logistic, run_direction_logistics,
)


class LabelRegressionTests(unittest.TestCase):
    def setUp(self):
        index = pd.MultiIndex.from_tuples(
            [(5, -0.5), (5, -0.4), (21, -0.5), (21, -0.4), (88, -0.5), (88, -0.4)],
            names=["bout_id", "bin_start_s"],
        )
        self.data = pd.DataFrame({"PC1": np.arange(6.0), "PC2": np.arange(6.0) ** 2}, index=index)
        self.labels = pd.DataFrame(
            {"position": 5 + 2 * self.data.PC1 - 3 * self.data.PC2, "unused_label": np.nan},
            index=index,
        )

    def test_known_linear_relationship_and_observation_identity(self):
        original_data, original_labels = self.data.copy(), self.labels.copy()
        result = fit_label_regression(self.data, self.labels, target="position")
        self.assertAlmostEqual(result["r2"], 1.0)
        self.assertAlmostEqual(result["mae"], 0.0)
        np.testing.assert_allclose(result["model"].coef_, [2, -3], atol=1e-12)
        self.assertAlmostEqual(result["model"].intercept_, 5.0)
        self.assertTrue(result["predictions"].index.equals(self.data.index))
        np.testing.assert_allclose(result["predictions"]["residual"], 0, atol=1e-12)
        self.assertEqual(result["n_excluded"], 0)  # Unused missing labels do not exclude rows.
        pd.testing.assert_frame_equal(self.data, original_data)
        pd.testing.assert_frame_equal(self.labels, original_labels)

    def test_missing_and_infinite_targets_only_exclude_their_own_rows(self):
        self.labels.iloc[1, 0] = np.nan
        self.labels.iloc[4, 0] = np.inf
        result = fit_label_regression(self.data, self.labels, target="position")
        self.assertEqual(result["n_excluded"], 2)
        self.assertTrue(result["predictions"].index.equals(self.data.index[[0, 2, 3, 5]]))
        self.assertAlmostEqual(result["r2"], 1.0)

    def test_misaligned_rows_are_not_silently_matched(self):
        with self.assertRaisesRegex(ValueError, "identical indices"):
            fit_label_regression(self.data, self.labels.iloc[::-1], target="position")

    def test_invalid_targets_have_explicit_errors(self):
        for values, message in (
            (["x"] * 6, "numeric"),
            ([1.0] * 6, "no variation"),
            ([1.0] + [np.nan] * 5, "At least two"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                fit_label_regression(self.data, self.labels.assign(position=values), target="position")

    def test_plot_uses_the_supplied_fit_without_altering_it(self):
        result = fit_label_regression(self.data, self.labels, target="position")
        fig, ax = plt.subplots()
        returned_fig, returned_ax = plot_label_regression(result, ax=ax, color="red")
        self.assertIs(returned_fig, fig)
        self.assertIs(returned_ax, ax)
        np.testing.assert_allclose(
            ax.collections[0].get_offsets(), result["predictions"][["observed", "fitted"]],
        )
        fig.canvas.draw()
        plt.close(fig)

    def test_adjusted_r2_and_residual_degrees_of_freedom(self):
        labels = self.labels.copy()
        labels["position"] += [0, 0.4, -0.3, 0.8, -0.5, 0.2]
        result = fit_label_regression(self.data, labels, target="position")
        self.assertEqual(result["n_features"], 2)
        self.assertEqual(result["df_residual"], 3)
        self.assertAlmostEqual(result["r2_adjusted"], 1 - (1 - result["r2"]) * 5 / 3)

    def test_rank_deficiency_and_saturated_fit(self):
        collinear = self.data.assign(PC2=self.data.PC1)
        result = fit_label_regression(collinear, self.labels, target="position")
        self.assertEqual(result["df_residual"], 4)  # Rank-one predictors plus intercept.
        self.assertAlmostEqual(result["r2_adjusted"], 1 - (1 - result["r2"]) * 5 / 4)
        saturated = fit_label_regression(self.data.iloc[:3], self.labels.iloc[:3], target="position")
        self.assertEqual(saturated["df_residual"], 0)
        self.assertTrue(np.isnan(saturated["r2_adjusted"]))


class BatchRegressionTests(unittest.TestCase):
    def setUp(self):
        condition_fits = {}
        for condition in (0, 1):
            bout_ids = np.arange(10, 70, 10) + condition * 100
            index = pd.MultiIndex.from_product(
                [bout_ids, [-0.2, -0.1]], names=["bout_id", "bin_start_s"]
            )
            x = np.repeat(np.arange(6.0), 2)
            destination = 10 + 2 * x + 0.5 * x ** 2
            condition_fits[condition] = {"results": pd.DataFrame({
                "PC1": x, "PC2": x ** 2,
                "bin_number": np.tile([0, 1], 6),
                "bin_end_s": np.tile([-0.1, 0.0], 6),
                "initial_position": destination - np.repeat([0, 3, 6, -7, 8, -9], 2),
                "final_position": destination,
            }, index=index)}
        self.cache = {
            "sessions": {7: condition_fits},
            "session_info": pd.DataFrame(
                {"animal": ["example"], "headstage": [0], "region": ["AC"], "session": [3]}, index=[7]
            ),
            "settings": {"N_BINS": 2},
        }

    def test_batch_pools_windows_and_keeps_conditions_and_bout_ids(self):
        bundle = run_label_regressions(
            self.cache, target="final_position", columns=["PC1", "PC2"], min_displacement=5
        )
        self.assertEqual(len(bundle["summary"]), 2)
        self.assertEqual(len(bundle["fits"]), 2)
        self.assertEqual(bundle["settings"]["window_mode"], "pooled")
        self.assertTrue((bundle["summary"]["n_windows"] == 2).all())
        self.assertTrue((bundle["summary"]["n_observations"] == 8).all())
        self.assertTrue((bundle["summary"]["n_bouts"] == 4).all())
        self.assertTrue((bundle["summary"]["n_filtered"] == 4).all())
        self.assertTrue((bundle["summary"]["df_residual"] == 5).all())
        for (idx, condition), fit in bundle["fits"].items():
            self.assertEqual(idx, 7)
            expected_ids = np.repeat(np.arange(30, 70, 10) + condition * 100, 2).tolist()
            self.assertEqual(fit["predictions"].index.get_level_values("bout_id").tolist(), expected_ids)
            self.assertEqual(fit["predictions"].index.get_level_values("bin_start_s").tolist(), [-0.2, -0.1] * 4)
            rows = self.cache["sessions"][idx][condition]["results"]
            selected = rows.loc[(rows.final_position - rows.initial_position).abs() > 5]
            direct = fit_label_regression(selected[["PC1", "PC2"]], selected, target="final_position")
            pd.testing.assert_frame_equal(fit["predictions"], direct["predictions"])
        stream = io.BytesIO()
        pd.to_pickle(bundle, stream)
        stream.seek(0)
        reloaded = pd.read_pickle(stream)
        pd.testing.assert_frame_equal(bundle["summary"], reloaded["summary"])
        pd.testing.assert_frame_equal(bundle["fits"][(7, 0)]["predictions"], reloaded["fits"][(7, 0)]["predictions"])

    def test_invalid_subset_is_recorded_without_stopping_other_fits(self):
        self.cache["sessions"][7][1]["results"]["final_position"] = 1.0
        bundle = run_label_regressions(self.cache, target="final_position", columns=["PC1", "PC2"])
        self.assertEqual(len(bundle["fits"]), 1)
        skipped = bundle["summary"].xs(1, level="condition")
        self.assertTrue((skipped["status"] == "skipped").all())
        self.assertTrue(skipped["reason"].str.contains("no variation").all())

    def test_missing_coordinates_and_empty_filters_are_reported(self):
        missing = run_label_regressions(self.cache, target="final_position", columns=["PC9"])
        self.assertFalse(missing["fits"])
        self.assertTrue(missing["summary"]["reason"].str.contains("Missing columns: PC9").all())
        empty = run_label_regressions(
            self.cache, target="final_position", columns=["PC1"], min_displacement=100
        )
        self.assertFalse(empty["fits"])
        self.assertTrue((empty["summary"]["status"] == "skipped").all())
        self.assertTrue((empty["summary"]["n_observations"] == 0).all())

    def test_summary_plot_uses_saved_scores(self):
        bundle = run_label_regressions(self.cache, target="final_position", columns=["PC1", "PC2"])
        fig, axes = plot_regression_summary(bundle["summary"])
        for ax in axes:
            self.assertEqual(len(ax.lines), 2)  # Mean and zero reference.
            np.testing.assert_allclose(ax.collections[0].get_offsets(), [[7, 1.0]])
            self.assertEqual(ax.get_xlabel(), "Session entry (idx)")
        fig.canvas.draw()
        plt.close(fig)


class PairedRegressionTests(unittest.TestCase):
    def setUp(self):
        # Deliberately interleave session/condition rows; pairing must use idx.
        self.summary = pd.DataFrame([
            (99, 1, 0.2, 0.1, "ok"), (7, 0, 0.5, 0.4, "ok"),
            (42, 0, 0.3, 0.2, "ok"), (99, 0, 0.4, 0.3, "ok"),
            (7, 1, 0.3, 0.15, "ok"), (42, 1, np.nan, np.nan, "ok"),
            (51, 0, 0.4, 0.2, "ok"), (51, 1, 0.5, 0.3, "skipped"),
        ], columns=["idx", "condition", "r2", "r2_adjusted", "status"]).set_index(["idx", "condition"])

    def test_pairs_use_session_ids_and_exclude_incomplete_or_failed_pairs(self):
        for score, differences in (("r2", [0.2, 0.2]), ("r2_adjusted", [0.25, 0.2])):
            fig, axes, paired = plot_paired_regressions(self.summary, score=score)
            self.assertEqual(paired.index.tolist(), [7, 99])
            np.testing.assert_allclose(paired["difference"], differences)
            for idx in paired.index:
                self.assertAlmostEqual(paired.at[idx, "tracking"], self.summary.at[(idx, 0), score])
                self.assertAlmostEqual(paired.at[idx, "playback"], self.summary.at[(idx, 1), score])
            np.testing.assert_allclose(axes[1].collections[0].get_offsets(), np.column_stack(([7, 99], differences)))
            fig.canvas.draw()
            plt.close(fig)

    def test_no_complete_pairs_is_an_explicit_error(self):
        only_tracking = self.summary.loc[self.summary.index.get_level_values("condition") == 0]
        with self.assertRaisesRegex(ValueError, "No complete finite"):
            plot_paired_regressions(only_tracking)


class PositionBiasTests(unittest.TestCase):
    def setUp(self):
        # Unequal window counts would bias the distribution if rows were counted.
        index = pd.MultiIndex.from_tuples(
            [(10, -0.5), (10, -0.4), (10, -0.3), (20, -0.5), (30, -0.5), (30, -0.4), (40, -0.5)],
            names=["bout_id", "bin_start_s"],
        )
        self.bundle = {
            "settings": {"target": "initial_position"},
            "summary": pd.DataFrame(
                {"r2": [0.5, 0.2], "r2_adjusted": [0.4, 0.1], "status": ["ok", "ok"]},
                index=pd.MultiIndex.from_tuples([(7, 0), (7, 1)], names=["idx", "condition"]),
            ),
            "fits": {
                (7, 0): {"predictions": pd.DataFrame({"observed": [0, 0, 0, 7, 14, 14, 28]}, index=index)},
                (7, 1): {"predictions": pd.DataFrame({"observed": [1, 1, 1, 2, 3, 3, 4]}, index=index)},
            },
        }

    def test_distribution_uses_unique_bouts_and_midpoint_belongs_to_right(self):
        summary = summarize_starting_position_bias(self.bundle)
        self.assertEqual(summary.at[(7, 0), "n_position_bouts"], 4)
        self.assertEqual(summary.at[(7, 0), "n_left"], 2)
        self.assertEqual(summary.at[(7, 0), "n_right"], 2)
        self.assertEqual(summary.at[(7, 0), "side_bias"], 0)
        self.assertEqual(summary.at[(7, 1), "side_bias"], -1)
        self.assertEqual(summary.at[(7, 1), "abs_side_bias"], 1)
        self.assertAlmostEqual(summary.at[(7, 0), "position_mean"], 12.25)
        self.assertAlmostEqual(summary.at[(7, 0), "position_std"], np.std([0, 7, 14, 28], ddof=1))
        self.assertNotIn("side_bias", self.bundle["summary"].columns)
        fig, _ = plot_position_bias_scores(summary)
        fig.canvas.draw()
        plt.close(fig)
        fig, ax, paired = plot_paired_bias_differences(summary)
        self.assertAlmostEqual(paired.at[7, "imbalance_difference"], -1)
        self.assertAlmostEqual(paired.at[7, "score_difference"], 0.3)
        np.testing.assert_allclose(ax.collections[0].get_offsets(), [[-1, 0.3]])
        plt.close(fig)

    def test_wrong_target_and_inconsistent_bout_labels_are_rejected(self):
        self.bundle["settings"]["target"] = "final_position"
        with self.assertRaisesRegex(ValueError, "initial_position"):
            summarize_starting_position_bias(self.bundle)
        self.bundle["settings"]["target"] = "initial_position"
        self.bundle["fits"][(7, 0)]["predictions"].iloc[1, 0] = 5
        with self.assertRaisesRegex(ValueError, "constant across windows"):
            summarize_starting_position_bias(self.bundle)


class DirectionLogisticTests(unittest.TestCase):
    def setUp(self):
        index = pd.MultiIndex.from_product(
            [range(10, 90, 10), [-0.2, -0.1]], names=["bout_id", "bin_start_s"]
        )
        x = np.repeat([-4, -3, -2, -1, 1, 2, 3, 4], 2)
        self.data = pd.DataFrame({"PC1": x * 1000.0, "PC2": x ** 2}, index=index)
        self.labels = pd.DataFrame({
            "initial_direction": np.sign(x).astype(float), "unused": np.nan,
            "bin_number": np.tile([0, 1], 8), "bin_end_s": np.tile([-0.1, 0.0], 8),
        }, index=index)
        rows = self.data.join(self.labels)
        self.cache = {
            "sessions": {7: {0: {"results": rows}, 1: {"results": rows.copy()}}},
            "session_info": pd.DataFrame(
                {"animal": ["example"], "headstage": [0], "region": ["AC"], "session": [3]}, index=[7]
            ),
            "settings": {"N_BINS": 2},
        }

    def test_direction_identity_probabilities_and_scaling_are_preserved(self):
        original = self.data.copy()
        result = fit_direction_logistic(self.data, self.labels)
        self.assertEqual(result["balanced_accuracy"], 1)
        self.assertEqual(result["recall_left"], 1)
        self.assertEqual(result["recall_right"], 1)
        predictions = result["predictions"]
        self.assertTrue(predictions.index.equals(self.data.index))
        np.testing.assert_array_equal(predictions.predicted, self.labels.initial_direction)
        np.testing.assert_allclose(predictions.probability_right, result["model"].predict_proba(self.data)[:, 1])
        self.assertTrue((predictions.loc[predictions.observed == 1, "probability_right"] > 0.5).all())
        self.assertEqual(result["model"].named_steps["logisticregression"].class_weight, "balanced")
        scaled = result["model"].named_steps["standardscaler"].transform(self.data)
        np.testing.assert_allclose(scaled.std(axis=0), 1)
        pd.testing.assert_frame_equal(original, self.data)

    def test_unknown_directions_are_excluded_and_both_classes_required(self):
        self.labels.iloc[1, 0] = 0
        self.labels.iloc[9, 0] = np.nan
        result = fit_direction_logistic(self.data, self.labels)
        self.assertEqual(result["n_excluded"], 2)
        self.assertTrue(result["predictions"].index.equals(self.data.drop(self.data.index[[1, 9]]).index))
        with self.assertRaisesRegex(ValueError, "Both left"):
            fit_direction_logistic(self.data, self.labels.assign(initial_direction=-1))
        with self.assertRaisesRegex(ValueError, "identical indices"):
            fit_direction_logistic(self.data, self.labels.iloc[::-1])
        with self.assertRaisesRegex(ValueError, "finite positive"):
            fit_direction_logistic(self.data, self.labels, C=0)

    def test_balanced_accuracy_does_not_reward_a_constant_majority_prediction(self):
        labels = self.labels.assign(initial_direction=[-1] * 14 + [1] * 2)
        result = fit_direction_logistic(self.data * 0, labels)
        self.assertEqual(result["predictions"].predicted.nunique(), 1)
        self.assertEqual(result["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(result["balanced_accuracy"], (result["recall_left"] + result["recall_right"]) / 2)

    def test_batch_pools_windows_keeps_counts_and_serializes(self):
        self.cache["sessions"][7][1]["results"].iloc[0, 2] = 0
        result = run_direction_logistics(self.cache, columns=["PC1", "PC2"])
        self.assertEqual(len(result["fits"]), 2)
        self.assertEqual(result["summary"].index.names, ["idx", "condition"])
        self.assertEqual(result["summary"].at[(7, 0), "n_bouts"], 8)
        self.assertEqual(result["summary"].at[(7, 0), "n_observations"], 16)
        self.assertEqual(result["summary"].at[(7, 0), "n_left_bouts"], 4)
        self.assertEqual(result["summary"].at[(7, 0), "n_left_observations"], 8)
        self.assertEqual(result["summary"].at[(7, 1), "n_excluded_target"], 1)
        self.assertEqual(result["settings"]["window_mode"], "pooled")
        self.assertTrue(result["settings"]["standardized"])
        stream = io.BytesIO()
        pd.to_pickle(result, stream)
        stream.seek(0)
        reloaded = pd.read_pickle(stream)
        pd.testing.assert_frame_equal(result["summary"], reloaded["summary"])
        np.testing.assert_array_equal(
            reloaded["fits"][(7, 0)]["model"].predict(self.data), result["fits"][(7, 0)]["predictions"].predicted
        )
        self.cache["sessions"][7][1]["results"]["initial_direction"] = -1
        skipped = run_direction_logistics(self.cache, columns=["PC1"])
        self.assertEqual(len(skipped["fits"]), 1)
        self.assertEqual(skipped["summary"].at[(7, 1), "status"], "skipped")
        self.assertIn("Both left", skipped["summary"].at[(7, 1), "reason"])
        missing = run_direction_logistics(self.cache, columns=["PC9"])
        self.assertFalse(missing["fits"])
        self.assertTrue(missing["summary"].reason.str.contains("Missing columns: PC9").all())

    def test_plots_use_saved_balanced_scores_and_match_session_ids(self):
        summary = pd.DataFrame([
            (99, 1, 0.6, "ok"), (7, 0, 0.8, "ok"), (99, 0, 0.9, "ok"),
            (7, 1, 0.5, "ok"), (42, 0, 0.8, "ok"), (42, 1, np.nan, "skipped"),
        ], columns=["idx", "condition", "balanced_accuracy", "status"]).set_index(["idx", "condition"])
        with patch("sklearn.linear_model.LogisticRegression.fit", side_effect=AssertionError("Unexpected fit")):
            fig, axes = plot_regression_summary(summary, score="balanced_accuracy")
            for ax in axes:
                self.assertEqual(ax.get_ylabel(), "Balanced accuracy")
                np.testing.assert_allclose(ax.lines[-1].get_ydata(), [0.5, 0.5])
            fig.canvas.draw()
            plt.close(fig)
            fig, axes, paired = plot_paired_regressions(summary, score="balanced_accuracy")
            self.assertEqual(paired.index.tolist(), [7, 99])
            np.testing.assert_allclose(paired.difference, [0.3, 0.3])
            np.testing.assert_allclose(axes[0].lines[-1].get_ydata(), [0.5, 0.5])
            np.testing.assert_allclose(axes[1].lines[0].get_ydata(), [0, 0])
            fig.canvas.draw()
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
