"""Measure a separate train-only L-BFGS fit without replacing the app model.

Run ``python -m ev_project.training_diagnostics`` from the project directory.
The native scikit-learn FitCallback receives a ready-to-predict estimator after
each completed L-BFGS iteration. Validation loss is observed only: it does not
change the optimizer, stopping rule, model selection, or application artifact.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import log_loss

from .modeling import (
    SEED, _fit_with_diagnostics, _json_value, _sha256, fit_diagnostics,
    make_candidates,
)
from .schema import FEATURE_COLUMNS, ID_COLUMN, TARGET_COLUMN, validate_profiles


class LossHistoryCallback:
    """A public FitCallback observer for actual binary L-BFGS iterations.

    This implements the FitCallback protocol without depending on private
    optimizer functions. The final empty callback task has no fitted_estimator
    and is deliberately excluded. Validation arrays live only in this temporary
    observer and callbacks are detached in ``trace_logistic_training``.
    """

    def __init__(self, validation_features, validation_target, *, started=None):
        self.validation_features = validation_features
        self.validation_target = np.asarray(validation_target, dtype=np.int8)
        self.started = time.perf_counter() if started is None else started
        self.rows = []

    def setup(self, estimator, context):
        pass

    def teardown(self, estimator, context):
        pass

    def on_fit_task_begin(self, estimator, context, *, metadata=None):
        pass

    def on_fit_task_end(self, estimator, context, *, X=None, y=None, metadata=None, fitted_estimator=None):
        if context.task_name != 'lbfgs-iter' or fitted_estimator is None:
            return False
        train_probabilities = fitted_estimator.predict_proba(X)
        validation_probabilities = fitted_estimator.predict_proba(self.validation_features)
        self.rows.append({
            'iteration': int(context.task_id) + 1,
            'training_log_loss': float(log_loss(y, train_probabilities, labels=fitted_estimator.classes_)),
            'validation_log_loss': float(log_loss(
                self.validation_target, validation_probabilities, labels=fitted_estimator.classes_,
            )),
            'elapsed_seconds': time.perf_counter() - self.started,
        })
        # An observer never requests early stopping or changes the estimator.
        return False


def _normalized_labeled(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame) == 0:
        raise ValueError('Diagnostic train and validation sets must both be nonempty.')
    target = pd.to_numeric(frame[TARGET_COLUMN], errors='raise')
    if target.isna().any() or not target.isin([0, 1]).all() or target.nunique() != 2:
        raise ValueError('Diagnostic labels must contain both binary classes 0 and 1.')
    normalized, _ = validate_profiles(frame.drop(columns=TARGET_COLUMN))
    normalized[TARGET_COLUMN] = target.to_numpy(dtype=np.int8)
    return normalized.reset_index(drop=True)


def trace_logistic_training(train: pd.DataFrame, validation: pd.DataFrame,
                            pipeline=None):
    """Return a fresh fitted pipeline and a real per-iteration loss history.

    There is deliberately no holdout argument. Preprocessing learns only from
    train; the validation matrix is transformed with those learned parameters.
    The return value is for diagnostics, not an application-model replacement.
    """
    train = _normalized_labeled(train)
    validation = _normalized_labeled(validation)
    if set(train[ID_COLUMN]).intersection(validation[ID_COLUMN]):
        raise ValueError('Diagnostic train and validation IDs must be disjoint.')
    fitted = clone(make_candidates()['Logistic Regression'] if pipeline is None else pipeline)
    classifier = fitted.named_steps['classifier']
    parameters = classifier.get_params(deep=False)
    if parameters.get('solver') != 'lbfgs' or not hasattr(classifier, 'set_callbacks'):
        raise RuntimeError('The diagnostic requires LogisticRegression with native lbfgs callbacks (scikit-learn 1.9 or newer).')
    # Import the public protocol here to keep ordinary application imports light.
    from sklearn.callback import FitCallback
    history = None
    started = time.perf_counter()

    def fit_operation():
        nonlocal history
        preprocessing = fitted.named_steps['preprocessor']
        train_features = preprocessing.fit_transform(train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN])
        validation_features = preprocessing.transform(validation[list(FEATURE_COLUMNS)])
        history = LossHistoryCallback(validation_features, validation[TARGET_COLUMN], started=started)
        if not isinstance(history, FitCallback):
            raise TypeError('LossHistoryCallback must implement the public FitCallback protocol.')
        classifier.set_callbacks(history)
        try:
            classifier.fit(train_features, train[TARGET_COLUMN])
        finally:
            # Never serialize validation data or callback state in a pipeline.
            classifier.set_callbacks()

    _fit_with_diagnostics(
        fitted, len(train), fit_operation, fit_scope='fresh_diagnostic_train_only_70_percent',
    )
    frame = pd.DataFrame(history.rows, columns=[
        'iteration', 'training_log_loss', 'validation_log_loss', 'elapsed_seconds',
    ])
    actual = fit_diagnostics(fitted)['actual_iterations']
    if actual is None or frame['iteration'].tolist() != list(range(1, actual + 1)):
        raise RuntimeError('Native callback history does not match the fitted optimizer n_iter_.')
    if not np.isfinite(frame[['training_log_loss', 'validation_log_loss', 'elapsed_seconds']]).all().all():
        raise RuntimeError('The diagnostic history must contain finite measurements.')
    return fitted, frame


def _frame_sha256(frame: pd.DataFrame) -> str:
    """Hash the exact ordered, normalized diagnostic inputs without saving IDs."""
    import hashlib
    canonical = frame[[ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]].to_csv(
        index=False, lineterminator='\n', float_format='%.17g',
    ).encode('utf-8')
    return hashlib.sha256(canonical).hexdigest()


def _load_train_validation(root: Path, metadata: dict):
    """Use saved membership; select train/validation before input normalization.

    Holdout profile values and labels are not validated, transformed, evaluated,
    or passed to the trace. Reading the CSV container or hashing its bytes does
    not expose holdout observations to the optimizer or curve calculations.
    """
    membership = pd.read_csv(root / 'data/splits/split_membership.csv', dtype={ID_COLUMN: 'string'})
    if list(membership.columns) != [ID_COLUMN, 'split'] or membership[ID_COLUMN].duplicated().any():
        raise ValueError('Saved split membership must have unique IDs and columns id, split.')
    if set(membership['split']) != {'train', 'validation', 'holdout'}:
        raise ValueError('Saved split membership must contain exactly train, validation, and holdout.')
    counts = membership['split'].value_counts().to_dict()
    if counts != metadata['split_counts']:
        raise ValueError('Saved split counts do not match application-model metadata.')
    selected = membership.loc[membership['split'].isin(['train', 'validation'])]
    # Chunk selection discards holdout observations before schema or label handling.
    wanted = set(selected[ID_COLUMN])
    chunks = []
    for chunk in pd.read_csv(root / 'data/processed/train_clean.csv', dtype={ID_COLUMN: 'string'}, chunksize=100_000):
        chunks.append(chunk.loc[chunk[ID_COLUMN].isin(wanted)])
    data = pd.concat(chunks, ignore_index=True)
    if data[ID_COLUMN].duplicated().any() or set(data[ID_COLUMN]) != wanted:
        raise ValueError('The labeled source must contain each selected training/validation ID once.')
    by_id = data.set_index(ID_COLUMN)
    frames = {}
    for name in ['train', 'validation']:
        ordered_ids = selected.loc[selected['split'] == name, ID_COLUMN]
        frames[name] = _normalized_labeled(by_id.loc[ordered_ids].reset_index())
    return frames['train'], frames['validation']


def plot_loss_history(history: pd.DataFrame, output: Path) -> Path:
    """Save an exportable loss figure with an explicit diagnostic-only scope."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False}):
        figure, axis = plt.subplots(figsize=(9.4, 5.6), constrained_layout=True)
        axis.plot(history['iteration'], history['training_log_loss'], label='Training (70%)', color='#146b8c', linewidth=2.2)
        axis.plot(history['iteration'], history['validation_log_loss'], label='Validation (15%)', color='#c06526', linewidth=2.2)
        axis.set(xlabel='Completed L-BFGS iteration', ylabel='Mean log loss (lower is better)')
        axis.set_title('Fresh diagnostic fit: training and validation loss', loc='left', pad=24, weight='bold')
        axis.text(0, 1.025, 'Train-fitted preprocessing · unchanged solver/settings · holdout excluded',
                  transform=axis.transAxes, fontsize=10, color='#4d5965')
        axis.grid(alpha=.18)
        axis.legend(frameon=False)
        axis.text(0, -.18, 'Observed prediction loss excludes the L2 penalty. This curve is not the historical app-model fit.',
                  transform=axis.transAxes, fontsize=9, color='#4d5965')
        figure.savefig(output, dpi=180, facecolor='white')
        plt.close(figure)
    return output


def run_training_diagnostics(root: Path) -> dict:
    """Write a fresh diagnostic report while keeping both model files identical."""
    root = Path(root).resolve()
    bundle_path = root / 'models/model_bundle.joblib'
    metadata_path = root / 'models/model_metadata.json'
    before = {'model_bundle.joblib': _sha256(bundle_path), 'model_metadata.json': _sha256(metadata_path)}
    bundle = joblib.load(bundle_path)
    metadata = bundle['metadata']
    if json.loads(metadata_path.read_text(encoding='utf-8')) != metadata:
        raise ValueError('Application bundle metadata and model_metadata.json disagree.')
    source_hashes = {}
    for name, relative in [('train_clean.csv', 'data/processed/train_clean.csv'), ('train.csv', 'train.csv')]:
        source_hashes[name] = _sha256(root / relative)
        if source_hashes[name] != metadata['source_sha256'][name]:
            raise ValueError(f'{name} no longer matches the source hash stored with the app model.')
    if metadata['selected_model'] != 'Logistic Regression':
        raise ValueError('The existing application model must be the prespecified Logistic Regression.')
    candidate = make_candidates()['Logistic Regression']
    saved_classifier = bundle['pipeline'].named_steps['classifier']
    if candidate.named_steps['classifier'].get_params(deep=False) != saved_classifier.get_params(deep=False):
        raise ValueError('Diagnostic settings must match the saved application classifier settings.')
    train, validation = _load_train_validation(root, metadata)
    print(f'Fresh diagnostic only: {len(train):,} training / {len(validation):,} validation rows; holdout excluded.', flush=True)
    fitted, history = trace_logistic_training(train, validation, candidate)
    historical = dict(
        model=metadata['selected_model'], fit_role='saved_application_model',
        **fit_diagnostics(bundle['pipeline'],
                          fit_rows=metadata['split_counts']['train'] + metadata['split_counts']['validation'],
                          fit_scope='historical_train_plus_validation_85_percent'),
    )
    fresh = dict(model='Logistic Regression', fit_role='fresh_train_only_diagnostic', **fit_diagnostics(fitted))
    historical_elapsed_recorded = historical['elapsed_seconds'] is not None
    historical_warning_recorded = historical['convergence_warning_observed'] is not None
    historical_measurement_note = (
        'Original elapsed fit time: ' + ('recorded' if historical_elapsed_recorded else 'not recorded') + '. '
        'Original warning observation: ' + ('recorded' if historical_warning_recorded else 'not recorded') + '. '
        'Missing measurements cannot be reconstructed. The original per-iteration loss history is not available; '
        'this report\'s curve comes from a separate train-only diagnostic refit.'
    )
    document = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Fresh train-only convergence diagnostic; historical application model is not refitted or replaced.',
        'fit_records': [historical, fresh],
        'loss_history': {
            'file': 'lbfgs_loss_history.csv', 'rows': len(history),
            'collection_method': 'Native public sklearn.callback.FitCallback via LogisticRegression.set_callbacks',
            'iteration_source': 'Completed lbfgs-iter context.task_id + 1; empty task with no fitted_estimator excluded; count checked against n_iter_.',
            'loss_definition': 'Mean unpenalized binary log loss, computed from the native fitted_estimator after each completed L-BFGS iteration.',
            'training_rows': len(train), 'validation_rows': len(validation),
            'holdout_used_for_fit_or_curve': False, 'validation_used_for_stopping': False,
            'callbacks_detached_after_fit': True, 'elapsed_includes_callback_scoring': True,
            'first_training_log_loss': float(history.iloc[0]['training_log_loss']),
            'final_training_log_loss': float(history.iloc[-1]['training_log_loss']),
            'first_validation_log_loss': float(history.iloc[0]['validation_log_loss']),
            'final_validation_log_loss': float(history.iloc[-1]['validation_log_loss']),
        },
        'iteration_interpretation': 'max_iter=1000 is an upper cap, not 1000 required iterations. lbfgs uses full-data objective/gradient evaluations; no minibatch size applies.',
        'termination_interpretation': 'The diagnostic fit completed without ConvergenceWarning. The public estimator API does not expose whether the exact optimizer termination criterion was gradient or function-value tolerance; n_iter_ below the cap alone does not establish the reason.',
        'historical_measurement_limit': historical_measurement_note,
        'saved_application_measurement_availability': {
            'elapsed_seconds_recorded': historical_elapsed_recorded,
            'convergence_warning_observation_recorded': historical_warning_recorded,
            'historical_loss_history_recorded': False,
        },
        'settings_unchanged': saved_classifier.get_params(deep=False),
        'provenance': {
            'seed': SEED, 'saved_model_trained_at_utc': metadata['trained_at_utc'],
            'source_sha256': source_hashes,
            'split_membership_sha256': _sha256(root / 'data/splits/split_membership.csv'),
            'ordered_normalized_input_sha256': {'train': _frame_sha256(train), 'validation': _frame_sha256(validation)},
            'application_artifact_sha256': before,
            'source_code_sha256': {'modeling.py': _sha256(Path(__file__).with_name('modeling.py')),
                                   'training_diagnostics.py': _sha256(Path(__file__))},
            'versions': {name: importlib.metadata.version(name) for name in ['scikit-learn', 'scipy', 'numpy', 'pandas', 'joblib']},
            'python_version': platform.python_version(),
        },
    }
    reports = root / 'docs/reports/modeling'
    reports.mkdir(parents=True, exist_ok=True)
    history.to_csv(reports / 'lbfgs_loss_history.csv', index=False)
    pd.DataFrame(document['fit_records']).to_csv(reports / 'training_diagnostics.csv', index=False)
    plot_loss_history(history, reports / 'figures/lbfgs_loss_history.png')
    after = {'model_bundle.joblib': _sha256(bundle_path), 'model_metadata.json': _sha256(metadata_path)}
    if after != before:
        raise RuntimeError('An application-model file changed during the diagnostic run.')
    document['provenance']['application_artifacts_byte_identical'] = True
    document['provenance']['output_sha256'] = {name: _sha256(reports / name) for name in [
        'lbfgs_loss_history.csv', 'training_diagnostics.csv', 'figures/lbfgs_loss_history.png',
    ]}
    (reports / 'training_diagnostics.json').write_text(json.dumps(document, indent=2, default=_json_value), encoding='utf-8')
    print(f"Recorded {fresh['actual_iterations']} real L-BFGS iterations in {fresh['elapsed_seconds']:.2f} seconds (including monitoring).", flush=True)
    print(f'Saved diagnostics to {reports}; application model files remain byte-identical.', flush=True)
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1], help='Project root (default: this package\'s project).')
    arguments = parser.parse_args()
    run_training_diagnostics(arguments.root)


if __name__ == '__main__':
    main()
