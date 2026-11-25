"""Match current speed targets to existing PCA windows by original bout ID."""

import unittest

import numpy as np
import pandas as pd

from evaluation import run_label_regressions


class CurrentBoutTargetTests(unittest.TestCase):
    def setUp(self):
        sessions, labels = {}, []
        for condition in (0, 1):
            ids = np.array([17, 4, 93, 28]) + 100 * condition
            index = pd.MultiIndex.from_product([ids, [-.2, -.1]], names=["bout_id", "bin_start_s"])
            sessions[condition] = {"results": pd.DataFrame({
                "PC1": np.repeat(np.arange(4.), 2), "PC2": np.tile([0., 1.], 4),
                "bin_number": np.tile([0, 1], 4), "bin_end_s": np.tile([-.1, 0.], 4),
                # Stale target values must be overridden, not silently reused.
                "early_mean_speed": 999., "peak_speed": 999.,
            }, index=index)}
            for number, bout_id in enumerate(ids):
                labels.append({
                    "bout_id": bout_id, "idx": 7, "condition": condition,
                    "early_mean_speed": 2. + 3 * number,
                    "peak_speed": 5. + 4 * number,
                    "initial_position": np.nan, "final_position": np.nan,
                })
        self.bouts = pd.DataFrame(labels).set_index("bout_id").iloc[::-1]
        self.cache = {
            "sessions": {7: sessions},
            "session_info": pd.DataFrame({"animal": ["test"], "headstage": [0], "session": [2]}, index=[7]),
            "settings": {"N_BINS": 2, "activity_mode": "raw"},
        }

    def run_target(self, target="early_mean_speed", **overrides):
        kwargs = dict(target=target, columns=["PC1", "PC2"], bout_labels=self.bouts)
        kwargs.update(overrides)
        return run_label_regressions(self.cache, **kwargs)

    def test_unsorted_bout_ids_repeat_current_targets_and_preserve_both_sources(self):
        before_labels = self.bouts.copy(deep=True)
        before_rows = {condition: fit["results"].copy(deep=True)
                       for condition, fit in self.cache["sessions"][7].items()}
        for target in ["early_mean_speed", "peak_speed"]:
            result = self.run_target(target)
            self.assertEqual(result["settings"]["target_source"], "current_bout_labels")
            self.assertTrue(result["summary"]["status"].eq("ok").all())
            self.assertTrue(result["summary"]["n_filtered"].eq(0).all())
            for (_, condition), fit in result["fits"].items():
                original_rows = before_rows[condition]
                expected = self.bouts.loc[original_rows.index.get_level_values("bout_id"), target].to_numpy()
                np.testing.assert_array_equal(fit["predictions"]["observed"], expected)
                self.assertTrue(fit["predictions"].index.equals(original_rows.index))
                self.assertAlmostEqual(fit["r2"], 1.)
        pd.testing.assert_frame_equal(before_labels, self.bouts)
        for condition, before in before_rows.items():
            pd.testing.assert_frame_equal(before, self.cache["sessions"][7][condition]["results"])

    def test_missing_speed_is_excluded_only_for_its_target_and_its_complete_bout(self):
        self.bouts.loc[4, "early_mean_speed"] = np.nan
        mean, peak = self.run_target(), self.run_target("peak_speed")
        self.assertEqual(mean["summary"].loc[(7, 0), "n_excluded_target"], 2)
        self.assertEqual(mean["summary"].loc[(7, 0), "n_bouts"], 3)
        self.assertEqual(peak["summary"].loc[(7, 0), "n_bouts"], 4)
        self.assertEqual(mean["summary"].loc[(7, 1), "n_bouts"], 4)
        self.assertNotIn(4, mean["fits"][(7, 0)]["predictions"].index.get_level_values("bout_id"))

    def test_missing_bout_ids_and_wrong_session_or_condition_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing saved bout IDs"):
            self.run_target(bout_labels=self.bouts.drop(index=17))
        for column, value in [("idx", 99), ("condition", 1)]:
            changed = self.bouts.copy()
            changed.loc[17, column] = value
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "session/condition"):
                self.run_target(bout_labels=changed)

    def test_ambiguous_or_incomplete_target_tables_are_rejected(self):
        duplicate = pd.concat([self.bouts, self.bouts.iloc[:1]])
        with self.assertRaisesRegex(ValueError, "unique row"):
            self.run_target(bout_labels=duplicate)
        with self.assertRaisesRegex(ValueError, "missing columns"):
            self.run_target(bout_labels=self.bouts.drop(columns="idx"))
        with self.assertRaises(TypeError):
            self.run_target(bout_labels=self.bouts["early_mean_speed"])

    def test_all_missing_current_target_is_skipped_instead_of_using_cached_values(self):
        self.bouts["early_mean_speed"] = np.nan
        result = self.run_target()
        self.assertTrue(result["summary"]["status"].eq("skipped").all())
        self.assertTrue(result["summary"]["n_observations"].eq(0).all())
        self.assertFalse(result["fits"])


if __name__ == "__main__":
    unittest.main()
