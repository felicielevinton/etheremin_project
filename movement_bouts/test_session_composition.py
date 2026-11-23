"""Check session-level associations, confounding controls and skipped designs."""

import unittest
import numpy as np
import pandas as pd

from evaluation import fit_session_composition_associations


class CompositionAssociationsTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        group = np.repeat([0, 1], 50)
        fraction = .15 + .55*group + rng.uniform(0, .15, 100)
        self.table = pd.DataFrame({"fraction": fraction, "score": 2*group - .4*fraction,
                                   "block": group.astype(str)}, index=pd.Index(np.arange(100)*3, name="idx"))

    def run_fit(self, table):
        return fit_session_composition_associations(
            table, score="score", fractions={"Motor": "fraction"},
            numeric_controls=[], categorical_controls=["block"],
        )

    def test_block_control_recovers_opposite_within_block_association(self):
        result = self.run_fit(self.table)
        row = result["summary"].loc["Motor"]
        self.assertEqual(row["status"], "ok")
        self.assertGreater(row["pearson_r"], 0)
        self.assertAlmostEqual(row["coefficient_per_10pp"], -.04, places=10)
        self.assertAlmostEqual(row["partial_r"], -1)
        self.assertGreater(row["added_session_r2"], 0)
        self.assertTrue(result["fits"]["Motor"]["observations"].index.equals(self.table.index))

    def test_missing_fraction_is_excluded_without_renumbering_sessions(self):
        table = self.table.copy()
        table.loc[9, "fraction"] = np.nan
        result = self.run_fit(table)
        self.assertEqual(result["summary"].loc["Motor", "n_excluded"], 1)
        self.assertEqual(result["summary"].loc["Motor", "n_sessions"], 99)
        self.assertNotIn(9, result["fits"]["Motor"]["observations"].index)

    def test_block_only_fraction_is_explicitly_skipped(self):
        table = self.table.copy()
        table["fraction"] = table["block"].astype(float)
        result = self.run_fit(table)
        self.assertEqual(result["summary"].loc["Motor", "status"], "skipped")
        self.assertIn("explained by controls", result["summary"].loc["Motor", "reason"])
        self.assertFalse(result["fits"])

    def test_repeated_session_rows_and_invalid_fractions_are_rejected(self):
        with self.assertRaises(ValueError):
            self.run_fit(pd.concat([self.table, self.table]))
        table = self.table.copy()
        table.loc[0, "fraction"] = 5
        with self.assertRaises(ValueError):
            self.run_fit(table)


if __name__ == "__main__":
    unittest.main()
