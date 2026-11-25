"""Early-speed interval boundaries and compatibility of enriched bout tables."""

import hashlib
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pandas as pd

from utils import detect_movement_bouts, save_movement_bout_tables, validate_bout_sources


class MovementBoutSpeedTests(unittest.TestCase):
    def detect(self, speed, **overrides):
        # The trajectory is only needed for legacy position/distance fields.
        features = pd.DataFrame({
            "Speed_x": np.asarray(speed, dtype=float),
            "Position": np.cumsum(speed, dtype=float) * .1,
            "Condition": np.zeros(len(speed)),
        })
        options = dict(dt=.1, speed_x_factor=1, velocity_threshold=.5,
                       min_immobility_s=.2, early_speed_window_s=1.)
        options.update(overrides)
        bouts, quiet = detect_movement_bouts(
            features, {"animal": "test", "headstage": 0, "session": 2}, **options
        )
        return bouts, quiet

    def test_one_second_cap_excludes_a_later_larger_peak(self):
        speeds = [1, 3, 5, 4, 2, 8, 1, 2, 3, 4, 99, 2]
        bouts, _ = self.detect([0] * 3 + speeds + [0] * 3)
        row = bouts.iloc[0]
        self.assertEqual(row.onset_bin, 3)
        self.assertEqual(row.end_bin, 15)
        self.assertAlmostEqual(row.early_mean_speed, np.mean(speeds[:10]))
        self.assertEqual(row.peak_speed, 8)
        self.assertAlmostEqual(row.peak_speed_latency_s, .5)
        self.assertEqual(row.early_speed_duration_s, 1.)
        self.assertTrue(row.early_speed_complete)
        self.assertAlmostEqual(row.mean_speed, np.mean(speeds))

    def test_earlier_qualifying_immobility_shortens_both_metrics(self):
        bouts, _ = self.detect([0] * 3 + [2, 6] + [0] * 3 + [100] * 12 + [0] * 3)
        row = bouts.iloc[0]
        self.assertEqual(row.early_mean_speed, 4)
        self.assertEqual(row.peak_speed, 6)
        self.assertAlmostEqual(row.early_speed_duration_s, .2)
        self.assertAlmostEqual(row.peak_speed_latency_s, .1)
        self.assertTrue(row.early_speed_complete)

    def test_short_pauses_retain_the_existing_bout_definition(self):
        bouts, _ = self.detect([0] * 3 + [2, 0, 4, 4] + [0] * 3)
        self.assertEqual(len(bouts), 1)
        self.assertEqual(bouts.iloc[0].early_mean_speed, 2.5)
        self.assertAlmostEqual(bouts.iloc[0].early_speed_duration_s, .4)

    def test_parameter_changes_only_early_fields_and_preserves_first_peak_tie(self):
        speed = [0] * 3 + [2, 7, 7, 4, 3, 9, 3, 2, 1, 1, 50] + [0] * 3
        short, quiet_short = self.detect(speed, early_speed_window_s=.3)
        longer, quiet_longer = self.detect(speed, early_speed_window_s=.8)
        legacy = [c for c in short if not c.startswith("early_") and not c.startswith("peak_")]
        pd.testing.assert_frame_equal(short[legacy], longer[legacy])
        pd.testing.assert_frame_equal(quiet_short, quiet_longer)
        self.assertEqual(short.iloc[0].peak_speed, 7)
        self.assertAlmostEqual(short.iloc[0].peak_speed_latency_s, .1)
        self.assertEqual(longer.iloc[0].peak_speed, 9)

    def test_unfinished_bout_has_speed_if_the_early_interval_is_observed(self):
        bouts, _ = self.detect([0] * 3 + [2] * 12)
        row = bouts.iloc[0]
        self.assertTrue(pd.isna(row.end_bin))
        self.assertTrue(pd.isna(row.mean_speed))
        self.assertEqual(row.early_mean_speed, 2)
        self.assertEqual(row.peak_speed, 2)
        self.assertEqual(row.peak_speed_latency_s, 0)
        self.assertTrue(row.early_speed_complete)

    def test_recording_truncation_keeps_the_bout_but_not_a_partial_speed_estimate(self):
        bouts, _ = self.detect([0] * 3 + [2] * 4)
        row = bouts.iloc[0]
        self.assertFalse(row.early_speed_complete)
        self.assertAlmostEqual(row.early_speed_duration_s, .4)
        for field in ["early_mean_speed", "peak_speed", "peak_speed_latency_s"]:
            self.assertTrue(pd.isna(row[field]))

    def test_no_movements_and_invalid_window_settings(self):
        bouts, _ = self.detect([0] * 20)
        self.assertTrue(bouts.empty)
        self.assertIn("early_mean_speed", bouts)
        for value in [0, -.1, np.nan, np.inf, .25]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.detect([0] * 3 + [2] * 12, early_speed_window_s=value)


class BoutSourceCompatibilityTests(unittest.TestCase):
    def setUp(self):
        test_root = Path(__file__).resolve().parent
        self.directory = test_root / f".test_bout_sources_{uuid4().hex}"
        self.directory.mkdir()

        def cleanup():
            if self.directory.resolve().parent != test_root:
                raise ValueError("Test cleanup must stay within the test directory.")
            shutil.rmtree(self.directory)

        self.addCleanup(cleanup)
        self.bouts = pd.DataFrame({"idx": [4, 4], "onset_bin": [30, 70],
                                   "initial_position": [2., 3.]}, index=[17, 9])
        self.quiet = pd.DataFrame({"start_bin": [0, 40]})
        self.sessions = pd.DataFrame({"animal": ["test"]}, index=[4])
        save_movement_bout_tables(self.bouts, self.quiet, self.sessions, {"version": 1}, self.directory)
        self.hashes = {name: hashlib.sha256((self.directory / name).read_bytes()).hexdigest()
                       for name in ["movement_bouts.pkl", "session_info.pkl"]}

    def enrich(self):
        result = self.bouts.assign(early_mean_speed=[2., 5.], peak_speed=[3., 7.])
        save_movement_bout_tables(result, self.quiet, self.sessions,
                                 {"EARLY_SPEED_WINDOW_S": 1.}, self.directory)
        return result

    def test_exact_and_column_extension_sources_validate_without_rewriting_other_tables(self):
        validate_bout_sources(self.directory, self.hashes)
        quiet_before = (self.directory / "immobility_bouts.pkl").read_bytes()
        enriched = self.enrich()
        validate_bout_sources(self.directory, self.hashes)
        self.assertEqual(hashlib.sha256((self.directory / "session_info.pkl").read_bytes()).hexdigest(),
                         self.hashes["session_info.pkl"])
        self.assertEqual((self.directory / "immobility_bouts.pkl").read_bytes(), quiet_before)
        pd.testing.assert_frame_equal(pd.read_pickle(self.directory / "movement_bouts.pkl"), enriched)
        archive = self.directory / "source_versions" / f"{self.hashes['movement_bouts.pkl']}.pkl"
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), self.hashes["movement_bouts.pkl"])
        self.assertEqual(json.loads(archive.with_suffix(".parameters.json").read_text()), {"version": 1})

    def test_changed_values_order_columns_or_index_dtype_are_rejected(self):
        enriched = self.enrich()
        variants = [enriched.assign(onset_bin=[31, 70]), enriched.iloc[::-1],
                    enriched.drop(columns="initial_position")]
        float_index = enriched.copy()
        float_index.index = float_index.index.astype(float)
        variants.append(float_index)
        for changed in variants:
            with self.subTest(changed=changed):
                changed.to_pickle(self.directory / "movement_bouts.pkl")
                with self.assertRaises(ValueError):
                    validate_bout_sources(self.directory, self.hashes)

    def test_corrupt_archive_and_changed_session_metadata_are_rejected(self):
        self.enrich()
        self.sessions.assign(animal="other").to_pickle(self.directory / "session_info.pkl")
        with self.assertRaises(ValueError):
            validate_bout_sources(self.directory, self.hashes)
        self.sessions.to_pickle(self.directory / "session_info.pkl")
        archive = self.directory / "source_versions" / f"{self.hashes['movement_bouts.pkl']}.pkl"
        archive.write_bytes(b"invalid")
        with self.assertRaises(ValueError):
            validate_bout_sources(self.directory, self.hashes)

    def test_notebook_constant_controls_detection_and_saved_parameters(self):
        notebook = json.loads(Path(__file__).with_name("analyse_movement_bouts.ipynb").read_text(encoding="utf-8"))
        speed = np.r_[np.zeros(240), np.full(50, 2.), np.full(50, 20.),
                      np.full(200, 2.), np.zeros(240)]
        namespace = {
            "pd": pd,
            "features": [pd.DataFrame({"Speed_x": speed / 2,
                                       "Position": np.cumsum(speed) * .005,
                                       "Condition": np.zeros(len(speed))})],
            "session_info": pd.DataFrame({"animal": ["test"], "headstage": [0], "session": [2]}),
            "data_dir": self.directory,
            "display": lambda *args: None,
        }
        exec("".join(notebook["cells"][5]["source"]), namespace)
        self.assertEqual(namespace["EARLY_SPEED_WINDOW_S"], 1.)
        namespace["EARLY_SPEED_WINDOW_S"] = .25
        exec("".join(notebook["cells"][6]["source"]), namespace)
        with patch("subprocess.check_output", side_effect=["testcommit\n", ""]):
            exec("".join(notebook["cells"][7]["source"]), namespace)
        results_dir = self.directory / "derived/movement_bouts"
        saved = pd.read_pickle(results_dir / "movement_bouts.pkl")
        self.assertEqual(saved.iloc[0].early_mean_speed, 2.)
        self.assertEqual(saved.iloc[0].peak_speed, 2.)
        self.assertEqual(saved.iloc[0].early_speed_duration_s, .25)
        self.assertEqual(json.loads((results_dir / "parameters.json").read_text())["EARLY_SPEED_WINDOW_S"], .25)


if __name__ == "__main__":
    unittest.main()
