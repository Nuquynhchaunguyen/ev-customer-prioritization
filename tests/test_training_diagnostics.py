"""Native iteration measurements, training isolation, and honest fit metadata."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
import tempfile
import unittest
import warnings

import joblib
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline

from ev_project import modeling
from ev_project.schema import FEATURE_COLUMNS, ID_COLUMN, TARGET_COLUMN
from ev_project import training_diagnostics as diagnostics
from test_modeling import labeled_profiles


class WarningClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, features, target):
        # An old iteration attribute must not be reported as this failed fit.
        self.n_iter_ = np.array([91])
        warnings.warn('The optimizer did not converge.', ConvergenceWarning)
        return self


class TrainingDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits = modeling.make_splits(labeled_profiles(1000))

    def test_native_history_matches_completed_iterations_and_final_predictions(self):
        fitted, history = diagnostics.trace_logistic_training(self.splits['train'], self.splits['validation'])
        classifier = fitted.named_steps['classifier']
        actual = int(classifier.n_iter_[0])
        self.assertGreater(actual, 1)
        self.assertEqual(history['iteration'].tolist(), list(range(1, actual + 1)))
        self.assertTrue(np.isfinite(history.select_dtypes('number')).all().all())
        self.assertTrue(history['elapsed_seconds'].is_monotonic_increasing)
        for name, column in [('train', 'training_log_loss'), ('validation', 'validation_log_loss')]:
            frame = self.splits[name]
            expected = log_loss(frame[TARGET_COLUMN], fitted.predict_proba(frame[FEATURE_COLUMNS]), labels=[0, 1])
            self.assertAlmostEqual(history.iloc[-1][column], expected, places=14)
        # A detached callback cannot carry validation arrays into a saved model.
        self.assertNotIn('_skl_callbacks', classifier.__dict__)
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'diagnostic.joblib'
            joblib.dump(fitted, filename)
            loaded = joblib.load(filename)
            self.assertNotIn('_skl_callbacks', loaded.named_steps['classifier'].__dict__)

    def test_native_observer_preserves_plain_fit_predictions_and_parameters(self):
        candidate = modeling.make_candidates()['Logistic Regression']
        expected_parameters = candidate.named_steps['classifier'].get_params()
        fitted, _ = diagnostics.trace_logistic_training(self.splits['train'], self.splits['validation'], candidate)
        plain = modeling._fit_checked(modeling.make_candidates()['Logistic Regression'], self.splits['train'])
        self.assertFalse(hasattr(candidate.named_steps['classifier'], 'coef_'))
        self.assertEqual(fitted.named_steps['classifier'].get_params(), expected_parameters)
        np.testing.assert_array_equal(
            plain.predict_proba(self.splits['validation'][FEATURE_COLUMNS]),
            fitted.predict_proba(self.splits['validation'][FEATURE_COLUMNS]),
        )
        np.testing.assert_array_equal(plain.named_steps['classifier'].n_iter_, fitted.named_steps['classifier'].n_iter_)

    def test_preprocessing_fits_train_only_and_validation_never_controls_optimizer(self):
        train = self.splits['train'].copy()
        train['Gender'] = 'Female'
        validation = self.splits['validation'].copy()
        validation['Gender'] = 'Other'
        first, first_history = diagnostics.trace_logistic_training(train, validation)
        preprocessing = first.named_steps['preprocessor']
        np.testing.assert_allclose(
            preprocessing.named_transformers_['numeric'].mean_,
            train[modeling.NUMERIC_FEATURES].mean().to_numpy(), atol=1e-12, rtol=0,
        )
        gender_index = modeling.CATEGORICAL_FEATURES.index('Gender')
        self.assertEqual(preprocessing.named_transformers_['categorical'].categories_[gender_index].tolist(), ['Female'])
        # Arbitrary validation changes affect its loss, but not the learned model.
        changed_validation = validation.copy()
        changed_validation[TARGET_COLUMN] = 1 - changed_validation[TARGET_COLUMN]
        changed_validation['Annual_Income_USD'] += 100_000
        second, second_history = diagnostics.trace_logistic_training(train, changed_validation)
        np.testing.assert_array_equal(first.named_steps['classifier'].coef_, second.named_steps['classifier'].coef_)
        np.testing.assert_array_equal(first_history['training_log_loss'], second_history['training_log_loss'])
        self.assertFalse(np.array_equal(first_history['validation_log_loss'], second_history['validation_log_loss']))
        self.assertNotIn('holdout', inspect.signature(diagnostics.trace_logistic_training).parameters)

    def test_trace_rejects_overlap_and_does_not_hide_convergence_warning(self):
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            diagnostics.trace_logistic_training(self.splits['train'], self.splits['train'])
        candidate = modeling.make_candidates()['Logistic Regression']
        candidate.set_params(classifier__max_iter=1)
        with self.assertRaises(ConvergenceWarning):
            diagnostics.trace_logistic_training(self.splits['train'], self.splits['validation'], candidate)

    def test_future_fit_measurements_and_unsupported_fields(self):
        fitted = modeling._fit_checked(
            modeling.make_candidates()['Logistic Regression'], self.splits['train'], fit_scope='unit_test_train',
        )
        details = modeling.fit_diagnostics(fitted)
        self.assertEqual(details['fit_rows'], 700)
        self.assertEqual(details['actual_iterations'], int(fitted.named_steps['classifier'].n_iter_[0]))
        self.assertEqual(details['n_iter_values'], fitted.named_steps['classifier'].n_iter_.tolist())
        self.assertEqual(details['max_iter'], 1000)
        self.assertEqual(details['tol'], 1e-4)
        self.assertEqual(details['fit_scope'], 'unit_test_train')
        self.assertGreater(details['elapsed_seconds'], 0)
        self.assertFalse(details['convergence_warning_observed'])
        self.assertFalse(details['iteration_limit_reached'])
        self.assertEqual(details['fit_status'], 'success')
        self.assertIn('exact optimizer termination reason is not exposed', details['optimizer_termination_note'])
        for name in ['Dummy baseline', 'Decision Tree']:
            pipeline = modeling._fit_checked(modeling.make_candidates()[name], self.splits['train'])
            unsupported = modeling.fit_diagnostics(pipeline)
            for field in ['actual_iterations', 'n_iter_values', 'max_iter', 'tol', 'iteration_limit_reached']:
                self.assertIsNone(unsupported[field], (name, field))
            self.assertFalse(unsupported['convergence_warning_observed'])
            self.assertGreater(unsupported['elapsed_seconds'], 0)

    def test_warnings_still_fail_and_record_the_observation(self):
        pipeline = Pipeline([('classifier', WarningClassifier())])
        with self.assertRaises(ConvergenceWarning):
            modeling._fit_checked(pipeline, self.splits['train'])
        measured = modeling.fit_diagnostics(pipeline)
        self.assertTrue(measured['convergence_warning_observed'])
        self.assertEqual(measured['fit_status'], 'failed_convergence_warning')
        self.assertIsNone(measured['actual_iterations'])
        self.assertGreater(measured['elapsed_seconds'], 0)

    def test_historical_snapshot_does_not_invent_elapsed_time_or_warning_observation(self):
        pipeline = modeling.make_candidates()['Logistic Regression']
        pipeline.fit(self.splits['train'][FEATURE_COLUMNS], self.splits['train'][TARGET_COLUMN])
        historical = modeling.fit_diagnostics(pipeline, fit_rows=700, fit_scope='historical')
        self.assertEqual(historical['actual_iterations'], int(pipeline.named_steps['classifier'].n_iter_[0]))
        self.assertEqual(historical['fit_status'], 'saved_artifact_snapshot')
        self.assertEqual(historical['fit_rows'], 700)
        self.assertIsNone(historical['elapsed_seconds'])
        self.assertIsNone(historical['convergence_warning_observed'])

    def test_saved_membership_loader_ignores_holdout_profile_values_and_labels(self):
        data = labeled_profiles(1000)
        splits = modeling.make_splits(data)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data/processed').mkdir(parents=True)
            (root / 'data/splits').mkdir(parents=True)
            membership = pd.concat([frame[[ID_COLUMN]].assign(split=name) for name, frame in splits.items()])
            membership.to_csv(root / 'data/splits/split_membership.csv', index=False)
            data.to_csv(root / 'data/processed/train_clean.csv', index=False)
            metadata = {'split_counts': {name: len(frame) for name, frame in splits.items()}}
            train, validation = diagnostics._load_train_validation(root, metadata)
            assert_frame_equal(train, splits['train'])
            assert_frame_equal(validation, splits['validation'])
            holdout_rows = data[ID_COLUMN].isin(splits['holdout'][ID_COLUMN])
            poisoned = data.copy()
            poisoned.loc[holdout_rows, 'Gender'] = 'not a valid category'
            poisoned[TARGET_COLUMN] = poisoned[TARGET_COLUMN].astype(object)
            poisoned.loc[holdout_rows, TARGET_COLUMN] = 'not a label'
            poisoned.to_csv(root / 'data/processed/train_clean.csv', index=False)
            unchanged_train, unchanged_validation = diagnostics._load_train_validation(root, metadata)
            assert_frame_equal(train, unchanged_train)
            assert_frame_equal(validation, unchanged_validation)

    def test_new_metadata_reports_measured_fit_details_on_future_training_save(self):
        splits = self.splits
        candidate = modeling._fit_checked(
            modeling.make_candidates()['Logistic Regression'], splits['train'],
            fit_scope='candidate_train_only_70_percent',
        )
        candidates = {'Logistic Regression': candidate}
        validation_metrics, validation_ranking, validation_predictions = modeling.evaluate_candidates(candidates, splits['validation'])
        final = modeling.fit_final_model(candidate, splits)
        holdout_metrics, holdout_ranking, holdout_predictions = modeling.evaluate_final(final, splits['holdout'], 'Logistic Regression')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'data/processed').mkdir(parents=True)
            data = pd.concat(list(splits.values()), ignore_index=True)
            data.to_csv(root / 'data/processed/train_clean.csv', index=False)
            data.to_csv(root / 'train.csv', index=False)
            test = labeled_profiles(10).drop(columns=TARGET_COLUMN)
            test[ID_COLUMN] = [f'test-{index}' for index in range(len(test))]
            test.to_csv(root / 'test.csv', index=False)
            reports = root / 'docs/reports/modeling'
            reports.mkdir(parents=True)
            previous_trace_files = {
                'training_diagnostics.json': b'{"existing_native_trace": true}',
                'training_diagnostics.csv': b'existing,native,trace\n',
                'lbfgs_loss_history.csv': b'iteration,training_log_loss,validation_log_loss\n1,0.3,0.31\n',
            }
            for filename, content in previous_trace_files.items():
                (reports / filename).write_bytes(content)
            metadata = modeling.save_training_artifacts(
                root, splits, candidates, validation_metrics, validation_ranking, validation_predictions,
                'Logistic Regression', final, holdout_metrics, holdout_ranking, holdout_predictions,
            )
            records = metadata['training_diagnostics']
            self.assertEqual([row['fit_rows'] for row in records], [700, 850])
            self.assertEqual([row['fit_role'] for row in records], ['candidate', 'final_application_model'])
            self.assertEqual([row['fit_scope'] for row in records], [
                'candidate_train_only_70_percent', 'final_train_plus_validation_85_percent',
            ])
            self.assertEqual(records[1]['actual_iterations'], int(final.named_steps['classifier'].n_iter_[0]))
            report = json.loads((reports / 'fit_summary.json').read_text())
            self.assertEqual(report['fit_records'], records)
            self.assertFalse(report['loss_history_recorded'])
            csv = pd.read_csv(reports / 'fit_summary.csv')
            self.assertEqual(csv['fit_rows'].tolist(), [700, 850])
            for filename, content in previous_trace_files.items():
                self.assertEqual((reports / filename).read_bytes(), content, filename)
            loaded_metadata = joblib.load(root / 'models/model_bundle.joblib')['metadata']
            self.assertEqual(loaded_metadata['training_diagnostics'], records)
            # A later diagnostic run must honor measurements saved by an
            # instrumented ordinary fit, rather than calling them unknown.
            artifact_hashes = {name: modeling._sha256(root / 'models' / name)
                               for name in ['model_bundle.joblib', 'model_metadata.json']}
            trace = diagnostics.run_training_diagnostics(root)
            saved_fit = trace['fit_records'][0]
            self.assertEqual(saved_fit['elapsed_seconds'], records[1]['elapsed_seconds'])
            self.assertFalse(saved_fit['convergence_warning_observed'])
            self.assertEqual(trace['saved_application_measurement_availability'], {
                'elapsed_seconds_recorded': True,
                'convergence_warning_observation_recorded': True,
                'historical_loss_history_recorded': False,
            })
            self.assertIn('Original elapsed fit time: recorded.', trace['historical_measurement_limit'])
            self.assertIn('Original warning observation: recorded.', trace['historical_measurement_limit'])
            self.assertIn('separate train-only diagnostic refit', trace['historical_measurement_limit'])
            self.assertEqual({name: modeling._sha256(root / 'models' / name)
                              for name in artifact_hashes}, artifact_hashes)


if __name__ == '__main__':
    unittest.main()
