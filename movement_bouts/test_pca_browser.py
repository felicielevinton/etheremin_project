"""Exercise the state a nested Plotly widget actually sends to the frontend."""

import json
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils import make_interactive_pca_browser, make_bout_pca_browser, make_start_end_position_browser


class BrowserDisplayTests(unittest.TestCase):
    def setUp(self):
        self.cache = {
            "sessions": {}, "position_limits": (0, 28),
            "session_info": pd.DataFrame({
                "animal": ["example", "example"], "headstage": [0, 0], "session": [0, 1],
            }, index=[0, 1]),
        }
        bouts = []
        for idx in (0, 1):
            self.cache["sessions"][idx] = {}
            for condition in (0, 1):
                bout_ids = [100 * idx + 10 * condition + 1, 100 * idx + 10 * condition + 2]
                index = pd.MultiIndex.from_product([bout_ids, [-0.2, -0.1]], names=["bout_id", "bin_start_s"])
                rows = pd.DataFrame({
                    "PC1": np.arange(4) + idx * 20, "PC2": [1, 3, 2, 4], "PC3": [3, 2, 4, 1],
                    "initial_position": [2, 2, 25, 25], "final_position": [8, 8, 19, 19],
                    "initial_direction": [-1, -1, 1, 1], "bin_number": [0, 1, 0, 1],
                    "bin_end_s": [-0.1, 0, -0.1, 0], "idx": idx, "condition": condition,
                }, index=index)
                model = SimpleNamespace(n_components_=3, explained_variance_ratio_=[0.6, 0.3, 0.1])
                self.cache["sessions"][idx][condition] = {"model": model, "results": rows}
                bouts.append(rows.reset_index().drop_duplicates("bout_id").set_index("bout_id"))
        self.bouts = pd.concat(bouts)
        self.browsers = []

    def tearDown(self):
        for browser in self.browsers:
            for child in browser.children:
                child.close()
            browser.close()

    def assert_display_ready(self, browser):
        self.browsers.append(browser)
        figure = browser.children[-1]
        state = figure.get_state()
        # A VBox only sends child model IDs; it never calls the child's repr.
        # Rebuild from the serialized initial state, not from figure.data.
        rendered = go.Figure(data=state["_widget_data"], layout=state["_widget_layout"])
        self.assertEqual(len(rendered.data), len(figure.data))
        self.assertTrue(any(len(trace.x) for trace in rendered.data))
        for actual, sent in zip(figure.data, rendered.data):
            self.assertEqual(list(actual.x), list(sent.x))
            self.assertEqual(list(actual.y), list(sent.y))
        self.assertEqual(rendered.layout.height, figure.layout.height)
        self.assertEqual(rendered.layout.width, figure.layout.width)
        self.assertEqual(rendered.layout.title.text, figure.layout.title.text)
        return figure

    def test_all_pca_color_and_filter_views_publish_populated_initial_state(self):
        for kwargs in ({"color_by": "position"}, {"color_by": "direction", "filter_position": True},
                       {"color_by": "destination"}, {"color_by": "destination", "filter_displacement": True}):
            with self.subTest(kwargs=kwargs):
                browser = make_interactive_pca_browser(self.cache, **kwargs)
                figure = self.assert_display_ready(browser)
                browser.children[0].value = 1
                self.assertIn("idx=1", figure.layout.title.text)
                self.assertTrue(all(x >= 20 for trace in figure.data for x in trace.x))
        tracking = {**self.cache, "projection_basis": "tracking"}
        self.assert_display_ready(make_interactive_pca_browser(tracking))

    def test_bout_and_endpoint_views_publish_populated_initial_state(self):
        browser = make_bout_pca_browser(self.cache)
        figure = self.assert_display_ready(browser)
        browser.children[1].value = 2
        self.assertEqual(list(figure.data[1].x), [2, 3])
        self.assert_display_ready(make_start_end_position_browser(self.bouts, self.cache["session_info"]))

    def test_linked_cameras_still_receive_frontend_relayout_events(self):
        figure = self.assert_display_ready(make_interactive_pca_browser(self.cache))
        camera = {"eye": {"x": 0.5, "y": 1.2, "z": 2.0}}
        figure._js2py_relayout = {"relayout_data": {"scene.camera": camera}, "source_view_id": "test-view"}
        self.assertEqual(figure.layout.scene.camera.to_plotly_json(), camera)
        self.assertEqual(figure.layout.scene2.camera.to_plotly_json(), camera)

    def test_frontend_trace_acknowledgements_accept_serialized_initial_data(self):
        browsers = [make_interactive_pca_browser(self.cache, color_by=color)
                    for color in ("position", "direction", "destination")]
        browsers.append(make_bout_pca_browser(self.cache))
        for browser in browsers:
            figure = self.assert_display_ready(browser)
            deltas = json.loads(go.Figure(figure).to_json(remove_uids=False))["data"]
            expected_x = [list(trace.x) for trace in figure.data]
            figure._js2py_traceDeltas = {
                "trace_deltas": deltas, "trace_edit_id": figure._last_trace_edit_id,
            }
            self.assertFalse(figure._trace_edit_in_process)
            self.assertEqual([list(trace.x) for trace in figure.data], expected_x)


if __name__ == "__main__":
    unittest.main()
