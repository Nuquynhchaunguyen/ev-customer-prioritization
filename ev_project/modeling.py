"""Fit and evaluate the three prespecified customer-prioritization baselines.

Model selection uses validation data only. The chosen estimator is refitted on
train + validation and that exact pipeline receives the final holdout evaluation
before it is saved for the application. Full-data descriptive EDA preceded this
experiment, so this is an internal evaluation, not external validation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .schema import (
    CATEGORICAL_FEATURES, FEATURE_COLUMNS, ID_COLUMN, NUMERIC_FEATURES,
    SCHEMA_VERSION, TARGET_COLUMN, validate_profiles,
)
from .scoring import rank_scores

SEED = 42
PRIMARY_CONTACT_FRACTION = 0.10
CONTACT_FRACTIONS = (0.01, 0.05, 0.10, 0.20)
MODEL_NAMES = ('Dummy baseline', 'Logistic Regression', 'Decision Tree')


def load_training_data(root: Path) -> pd.DataFrame:
    """Read conservative EDA output and apply the shared input normalization."""
    path = Path(root) / 'data/processed/train_clean.csv'
    if not path.is_file():
        raise FileNotFoundError('Run scripts/run_eda.py first to create data/processed/train_clean.csv.')
    data = pd.read_csv(path, dtype={ID_COLUMN: 'string'})
    if TARGET_COLUMN not in data:
        raise ValueError('The labeled training file must contain Will_Buy_EV.')
    target = pd.to_numeric(data[TARGET_COLUMN], errors='raise')
    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError('Clean training labels must be exactly No=0 or Yes=1.')
    if target.nunique() != 2:
        raise ValueError('Training data must contain both target classes.')
    normalized, _ = validate_profiles(data.drop(columns=TARGET_COLUMN))
    normalized[TARGET_COLUMN] = target.to_numpy(dtype=np.int8)
    return normalized.reset_index(drop=True)


def make_splits(data: pd.DataFrame, seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Create fixed 70/15/15 stratified membership, independent of scoring."""
    train, remaining = train_test_split(
        data, test_size=0.30, stratify=data[TARGET_COLUMN], random_state=seed,
    )
    validation, holdout = train_test_split(
        remaining, test_size=0.50, stratify=remaining[TARGET_COLUMN], random_state=seed + 1,
    )
    splits = {name: frame.reset_index(drop=True) for name, frame in
              [('train', train), ('validation', validation), ('holdout', holdout)]}
    memberships = [set(frame[ID_COLUMN]) for frame in splits.values()]
    assert all(not memberships[i].intersection(memberships[j])
               for i in range(3) for j in range(i + 1, 3)), 'Split IDs overlap.'
    assert sum(len(frame) for frame in splits.values()) == len(data)
    assert set.union(*memberships) == set(data[ID_COLUMN])
    return splits


def split_summary(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    total = sum(map(len, splits.values()))
    return pd.DataFrame([
        {'split': name, 'records': len(frame), 'fraction': len(frame) / total,
         'yes_count': int(frame[TARGET_COLUMN].sum()), 'yes_rate': float(frame[TARGET_COLUMN].mean())}
        for name, frame in splits.items()
    ])


def _preprocessor(scale_numeric: bool) -> ColumnTransformer:
    # Environmental concern is a category, despite its numeric storage. No
    # equal spacing or low-to-high survey meaning is inferred from its codes.
    return ColumnTransformer([
        ('numeric', StandardScaler() if scale_numeric else 'passthrough', list(NUMERIC_FEATURES)),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=True), list(CATEGORICAL_FEATURES)),
    ], remainder='drop', sparse_threshold=1.0, verbose_feature_names_out=True)


def make_candidates() -> dict[str, Pipeline]:
    """Prespecified settings, with no holdout-dependent tuning or class weights."""
    return {
        'Dummy baseline': Pipeline([('preprocessor', _preprocessor(False)),
                                   ('classifier', DummyClassifier(strategy='prior'))]),
        'Logistic Regression': Pipeline([('preprocessor', _preprocessor(True)),
                                       ('classifier', LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=SEED))]),
        'Decision Tree': Pipeline([('preprocessor', _preprocessor(False)),
                                 ('classifier', DecisionTreeClassifier(max_depth=6, min_samples_leaf=200, random_state=SEED))]),
    }


def _fit_checked(pipeline: Pipeline, data: pd.DataFrame) -> Pipeline:
    # Treat a convergence warning as a failed fit rather than quietly saving it.
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        pipeline.fit(data[list(FEATURE_COLUMNS)], data[TARGET_COLUMN])
    return pipeline


def fit_candidates(candidates: dict[str, Pipeline], train: pd.DataFrame) -> dict[str, Pipeline]:
    for name, pipeline in candidates.items():
        started = time.perf_counter()
        print(f'Fitting {name} on {len(train):,} training records...', flush=True)
        _fit_checked(pipeline, train)
        print(f'Completed {name} in {time.perf_counter() - started:.1f} seconds.', flush=True)
    return candidates


def _positive_probability(pipeline: Pipeline, data: pd.DataFrame) -> np.ndarray:
    classes = np.asarray(pipeline.classes_)
    index = np.flatnonzero(classes == 1)
    if len(index) != 1:
        raise ValueError('Expected the positive class 1 in the fitted pipeline.')
    probabilities = pipeline.predict_proba(data[list(FEATURE_COLUMNS)])[:, int(index[0])]
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('Predicted probabilities must be finite and between zero and one.')
    return probabilities


def _evaluate(name: str, pipeline: Pipeline, data: pd.DataFrame, split: str):
    y = data[TARGET_COLUMN].to_numpy(dtype=np.int8)
    probabilities = _positive_probability(pipeline, data)
    yes_rate = float(y.mean())
    ranking = rank_scores(data[ID_COLUMN], probabilities)
    labels_by_id = pd.Series(y, index=data[ID_COLUMN].astype(str))
    ordered_y = ranking[ID_COLUMN].map(labels_by_id).to_numpy(dtype=np.int8)
    positive_total = int(y.sum())
    rows = []
    for fraction in CONTACT_FRACTIONS:
        k = max(1, int(np.ceil(fraction * len(data))))
        yes_in_shortlist = int(ordered_y[:k].sum())
        precision = yes_in_shortlist / k
        rows.append({'model': name, 'split': split, 'contact_fraction': fraction, 'k': k,
                     'precision_at_k': precision, 'recall_at_k': yes_in_shortlist / positive_total,
                     'lift_at_k': precision / yes_rate, 'expected_random_precision': yes_rate,
                     'expected_random_lift': 1.0})
    ranking_metrics = pd.DataFrame(rows)
    primary = ranking_metrics.loc[ranking_metrics.contact_fraction == PRIMARY_CONTACT_FRACTION].iloc[0]
    at_05 = probabilities >= 0.5
    summary = {'model': name, 'split': split, 'n': len(data), 'yes_rate': yes_rate,
               'roc_auc': float(roc_auc_score(y, probabilities)),
               'average_precision': float(average_precision_score(y, probabilities)),
               'brier_score': float(brier_score_loss(y, probabilities)),
               'accuracy_at_05': float(accuracy_score(y, at_05)),
               'precision_at_05': float(precision_score(y, at_05, zero_division=0)),
               'recall_at_05': float(recall_score(y, at_05, zero_division=0)),
               'primary_k': int(primary['k']), 'precision_at_k': float(primary['precision_at_k']),
               'recall_at_k': float(primary['recall_at_k']), 'lift_at_k': float(primary['lift_at_k'])}
    predictions = pd.DataFrame({'id': data[ID_COLUMN].astype(str), 'y_true': y,
                                'model': name, 'yes_probability': probabilities, 'split': split})
    return pd.DataFrame([summary]), ranking_metrics, predictions


def evaluate_candidates(candidates: dict[str, Pipeline], validation: pd.DataFrame):
    outputs = [_evaluate(name, pipeline, validation, 'validation') for name, pipeline in candidates.items()]
    return tuple(pd.concat([result[index] for result in outputs], ignore_index=True) for index in range(3))


def select_model(validation_metrics: pd.DataFrame) -> str:
    """Maximize lift at 10% capacity, then ROC-AUC, then simpler baseline."""
    if not (validation_metrics['split'] == 'validation').all():
        raise ValueError('Model selection accepts validation results only.')
    complexity = {name: index for index, name in enumerate(MODEL_NAMES)}
    ordered = validation_metrics.assign(_complexity=validation_metrics.model.map(complexity)).sort_values(
        ['lift_at_k', 'roc_auc', '_complexity'], ascending=[False, False, True], kind='stable',
    )
    return str(ordered.iloc[0]['model'])


def fit_final_model(selected_pipeline: Pipeline, splits: dict[str, pd.DataFrame]) -> Pipeline:
    data = pd.concat([splits['train'], splits['validation']], ignore_index=True)
    print(f'Refitting the validation-selected model on {len(data):,} train + validation records...', flush=True)
    return _fit_checked(clone(selected_pipeline), data)


def evaluate_final(pipeline: Pipeline, holdout: pd.DataFrame, model_name: str):
    """Final evaluation of the exact pipeline that will be saved for scoring."""
    return _evaluate(model_name, pipeline, holdout, 'holdout')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Cannot serialize {type(value)}')


def save_training_artifacts(root: Path, splits, candidates, validation_metrics,
                            validation_ranking, validation_predictions, selected_model,
                            final_pipeline, holdout_metrics, holdout_ranking, holdout_predictions):
    root = Path(root)
    models = root / 'models'
    reports = root / 'reports/modeling'
    splits_dir = root / 'data/splits'
    demo_dir = root / 'data/demo'
    for directory in [models, reports / 'figures', splits_dir, demo_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    for name, frame in [('validation_metrics', validation_metrics), ('validation_ranking', validation_ranking),
                        ('validation_predictions', validation_predictions), ('holdout_metrics', holdout_metrics),
                        ('holdout_ranking', holdout_ranking), ('holdout_predictions', holdout_predictions),
                        ('split_summary', split_summary(splits))]:
        frame.to_csv(reports / f'{name}.csv', index=False)
    membership = pd.concat([frame[[ID_COLUMN]].assign(split=name) for name, frame in splits.items()], ignore_index=True)
    membership.to_csv(splits_dir / 'split_membership.csv', index=False)
    final_training = pd.concat([splits['train'], splits['validation']], ignore_index=True)
    training_ranges = {column: {'min': float(final_training[column].min()), 'max': float(final_training[column].max())}
                       for column in NUMERIC_FEATURES}
    metadata = {
        'schema_version': SCHEMA_VERSION, 'selected_model': selected_model,
        'trained_at_utc': datetime.now(timezone.utc).isoformat(), 'seed': SEED,
        'primary_contact_fraction': PRIMARY_CONTACT_FRACTION, 'contact_fractions': list(CONTACT_FRACTIONS),
        'selection_rule': 'Validation lift at 10% capacity, then ROC-AUC, then simpler model',
        'final_fit': 'Train + validation only (85%). Holdout excluded from fitting.',
        'evaluation_scope': 'Internal evaluation of synthetic Will_Buy_EV labels. Full-data descriptive EDA preceded splitting.',
        'feature_columns': list(FEATURE_COLUMNS), 'numeric_features': list(NUMERIC_FEATURES),
        'categorical_features': list(CATEGORICAL_FEATURES), 'id_used_by_model': False,
        'target_mapping': {'No': 0, 'Yes': 1}, 'training_ranges': training_ranges,
        'split_counts': {name: len(frame) for name, frame in splits.items()},
        'validation_metrics': validation_metrics.to_dict(orient='records'),
        'holdout_metrics': holdout_metrics.iloc[0].to_dict(),
        'holdout_ranking': holdout_ranking.to_dict(orient='records'),
        'model_parameters': {name: pipeline.named_steps['classifier'].get_params(deep=False)
                             for name, pipeline in candidates.items()},
        'versions': {name: importlib.metadata.version(name) for name in
                     ['scikit-learn', 'pandas', 'numpy', 'joblib', 'streamlit']},
        'python_version': platform.python_version(),
        'source_sha256': {'train_clean.csv': _sha256(root / 'data/processed/train_clean.csv'),
                          'train.csv': _sha256(root / 'train.csv')},
        'tie_rule': 'Descending probability. Tied scores use SHA256(seed=42, id), independent of upload row order.',
        'limitations': [
            'Synthetic profiles and supplied labels do not validate real customer purchases or revenue.',
            'Purchase timeframe, environmental concern survey anchors, and charging radius remain unknown.',
            'One internal split cannot establish generalization to a new customer population.',
            'No causal conclusions follow from predictor associations or model parameters.',
        ],
    }
    payload = {'schema_version': SCHEMA_VERSION, 'pipeline': final_pipeline, 'metadata': metadata}
    joblib.dump(payload, models / 'model_bundle.joblib', compress=3)
    joblib.dump(final_pipeline, models / 'selected_pipeline.joblib', compress=3)
    (models / 'model_metadata.json').write_text(json.dumps(metadata, indent=2, default=_json_value), encoding='utf-8')
    loaded = joblib.load(models / 'model_bundle.joblib')
    probe = splits['holdout'].iloc[:100]
    np.testing.assert_allclose(_positive_probability(final_pipeline, probe),
                               _positive_probability(loaded['pipeline'], probe), rtol=0, atol=0)
    # Scoring examples come exclusively from unlabeled Kaggle test profiles.
    test = pd.read_csv(root / 'test.csv', nrows=500, dtype={ID_COLUMN: 'string'})
    demo, _ = validate_profiles(test)
    demo.to_csv(demo_dir / 'customer_profiles_demo.csv', index=False)
    demo.iloc[:0].to_csv(demo_dir / 'customer_profiles_template.csv', index=False)
    # The 0.5 threshold is a diagnostic; shortlist membership depends on K.
    cm = confusion_matrix(holdout_predictions.y_true, holdout_predictions.yes_probability >= .5, labels=[0, 1])
    pd.DataFrame(cm, index=['actual_No', 'actual_Yes'], columns=['predicted_No', 'predicted_Yes']).to_csv(reports / 'holdout_confusion_matrix.csv')
    (reports / 'methodology.json').write_text(json.dumps({
        'selection_rule': metadata['selection_rule'], 'seed': SEED,
        'contact_fractions': list(CONTACT_FRACTIONS), 'split_summary': split_summary(splits).to_dict(orient='records'),
        'final_fit': metadata['final_fit'], 'tie_rule': metadata['tie_rule'],
        'holdout_used_for_selection': False,
    }, indent=2), encoding='utf-8')
    print(f'Saved model bundle, metadata, split membership, evaluation tables, and demo CSV under {root}.', flush=True)
    return metadata


def run_training(root: Path):
    """Terminal alternative to running the teaching notebook."""
    data = load_training_data(root)
    splits = make_splits(data)
    candidates = fit_candidates(make_candidates(), splits['train'])
    validation_metrics, validation_ranking, validation_predictions = evaluate_candidates(candidates, splits['validation'])
    selected = select_model(validation_metrics)
    print(f'Validation selected: {selected}', flush=True)
    final_pipeline = fit_final_model(candidates[selected], splits)
    holdout_metrics, holdout_ranking, holdout_predictions = evaluate_final(final_pipeline, splits['holdout'], selected)
    metadata = save_training_artifacts(root, splits, candidates, validation_metrics, validation_ranking,
        validation_predictions, selected, final_pipeline, holdout_metrics, holdout_ranking, holdout_predictions)
    print(holdout_metrics.to_string(index=False), flush=True)
    return metadata
