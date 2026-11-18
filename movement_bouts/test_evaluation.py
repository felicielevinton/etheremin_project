"""Checks for descriptive label regression; run with unittest discovery."""

import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation import fit_label_regression, plot_label_regression


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


if __name__ == "__main__":
    unittest.main()
