"""Check the new notebook's type matching, session cutoff and PCA wiring."""

import contextlib
import ast
import io
import json
from pathlib import Path
from unittest.mock import patch
import unittest

import numpy as np
import pandas as pd

from utils import prepare_pca_input, fit_pca_by_condition


CELL_SOURCES = [
    ''.join(cell['source'])
    for cell in json.loads(Path(__file__).with_name('neuron_type_pca.ipynb').read_text(encoding='utf-8'))['cells']
    if cell['cell_type'] == 'code'
]


def run_cell(cell_id, scope):
    # nbstripout renumbers cell IDs. Locate the actual code by a unique top-level
    # assignment so these checks also work against a freshly checked-out notebook.
    assignment = {'select-sessions': 'category_to_cluster', 'fit-pcas': 'run_settings'}[cell_id]
    matches = [source for source in CELL_SOURCES if any(
        isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == assignment for target in node.targets
        ) for node in ast.parse(source).body
    )]
    if len(matches) != 1:
        raise AssertionError(f"Expected one notebook cell assigning {assignment}, found {len(matches)}.")
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(matches[0], cell_id, 'exec'), scope)


class NeuronTypePcaTests(unittest.TestCase):
    def make_scope(self):
        return dict(
            pd=pd, np=np, display=lambda *args: None,
            NEURON_TYPE='Sensorimotor', MIN_NEURONS=5, ACTIVITY_MODE='raw',
            session_info=pd.DataFrame({'animal': ['A', 'B', 'C'], 'headstage': [0, 0, 0],
                                       'session': [0, 0, 0]}, index=[11, 27, 43]),
            neuron_ids_by_session=pd.DataFrame({'neuron_ids': [
                ['s1', 's2', 's3', 's4'],
                ['s7', 'missing', 's2', 's1', 's9', 's3'],
                ['n1', 'n2', 'n3', 'n4', 'n5', 'missing'],
            ]}, index=[11, 27, 43]),
            classification_table=pd.DataFrame({
                'neuron_idx': ['s1', 's2', 's3', 's4', 's7', 's9', 'n1', 'n2', 'n3', 'n4', 'n5'],
                'cluster_encoding': [2] * 6 + [3] * 5,
            }),
        )

    def test_exact_ids_original_order_and_inclusive_cutoff(self):
        scope = self.make_scope()
        run_cell('select-sessions', scope)
        self.assertEqual(scope['eligible_sessions'].index.tolist(), [27])
        self.assertEqual(scope['selected_neuron_ids'][27], ['s7', 's2', 's1', 's9', 's3'])
        self.assertEqual(scope['session_selection'].loc[27, 'n_unclassified'], 1)
        self.assertFalse(scope['session_selection'].loc[11, 'eligible'])

    def test_none_is_not_an_unclassified_unit(self):
        scope = self.make_scope()
        scope['NEURON_TYPE'] = 'None'
        run_cell('select-sessions', scope)
        self.assertEqual(scope['eligible_sessions'].index.tolist(), [43])
        self.assertEqual(scope['selected_neuron_ids'][43], ['n1', 'n2', 'n3', 'n4', 'n5'])

    def test_empty_selection_and_duplicate_classifications_fail_clearly(self):
        scope = self.make_scope()
        scope['NEURON_TYPE'] = 'Sensory'
        with self.assertRaisesRegex(ValueError, 'No sessions'):
            run_cell('select-sessions', scope)
        scope = self.make_scope()
        scope['classification_table'] = pd.concat([scope['classification_table']] * 2)
        with self.assertRaisesRegex(ValueError, 'unique'):
            run_cell('select-sessions', scope)

    def test_pca_uses_only_selected_columns_in_both_activity_modes(self):
        import hashlib
        import sklearn
        from datetime import datetime, timezone
        from numpy.testing import assert_allclose
        rng = np.random.default_rng(8)
        arrays = [rng.poisson(1, (6, 240)).astype(float) for _ in range(4)]
        ids = ['s7', 'missing', 's2', 's1', 's9', 's3']
        root = Path(__file__).resolve().parent
        classification_path = Path(__file__).resolve()
        for mode in ['raw', 'baseline_subtracted']:
            with self.subTest(mode=mode):
                scope = self.make_scope()
                scope.update(
                    Path=Path, hashlib=hashlib, sklearn=sklearn, datetime=datetime, timezone=timezone,
                    prepare_pca_input=prepare_pca_input, fit_pca_by_condition=fit_pca_by_condition,
                    ACTIVITY_MODE=mode, ANALYSIS_WINDOW_S=(-.5, 0), BASELINE_WINDOW_S=(-1, -.7),
                    DT=.005, N_BINS=5, extraction_dir=root, classification_path=classification_path,
                    extraction_parameters={'window_s': [-1.2, 0]},
                    movement_bouts=pd.DataFrame({'idx': [27]*4, 'condition': [0, 0, 1, 1],
                                                 'initial_position': [1, 2, 3, 4]}, index=[10, 20, 30, 40]),
                )
                run_cell('select-sessions', scope)
                with patch('pandas.read_pickle', return_value=pd.DataFrame(
                    {'spike_counts': arrays}, index=[10, 20, 30, 40]
                )):
                    run_cell('fit-pcas', scope)
                expected_ids = ['s7', 's2', 's1', 's9', 's3']
                self.assertEqual(scope['session_input'].columns.tolist(), expected_ids)
                for fit in scope['pca_cache']['sessions'][27].values():
                    self.assertEqual(fit['model'].feature_names_in_.tolist(), expected_ids)
                expected = arrays[0][:, 140:240].reshape(6, 5, 20).sum(-1) / .1
                if mode == 'baseline_subtracted':
                    expected -= arrays[0][:, 40:100].sum(-1)[:, None] / .3
                selected_indices = [ids.index(neuron) for neuron in expected_ids]
                assert_allclose(scope['session_input'].loc[10].to_numpy(), expected[selected_indices].T)
                self.assertEqual(scope['session_labels'].index.names, ['bout_id', 'bin_start_s'])
                self.assertEqual(scope['session_labels'].index.get_level_values('bout_id').unique().tolist(), [10, 20, 30, 40])


if __name__ == '__main__':
    unittest.main()
