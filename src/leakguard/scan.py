"""`scan()`: run every check on a dataframe and collect the findings."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import checks
from .report import Report


def scan(
    df: pd.DataFrame,
    target: str,
    *,
    time_col: str | None = None,
    id_cols: Iterable[str] = (),
    test_df: pd.DataFrame | None = None,
    top_k: int = 5,
    max_rows: int = 20_000,
    n_splits: int = 5,
    random_state: int = 0,
) -> Report:
    """Look for data leakage in a classification dataset.

    Parameters
    ----------
    df : the training data, one row per example.
    target : name of the column you want to predict (binary or multiclass).
    time_col : optional date/time column. Enables the train-on-past, test-on-future check.
    id_cols : columns you already know are identifiers. They are checked but not modelled.
    test_df : optional separate test set. Enables the train/test overlap check.
    top_k : how many of the strongest columns get the (slower) drop-column test.
    max_rows : larger datasets are randomly sampled down to this many rows (stratified).
    n_splits : folds for cross-validation.
    random_state : seed, for repeatable results.
    """
    if target not in df.columns:
        raise ValueError(f"target column '{target}' is not in the dataframe")
    id_cols = [c for c in id_cols]
    for col in [*id_cols, *([time_col] if time_col else [])]:
        if col not in df.columns:
            raise ValueError(f"column '{col}' is not in the dataframe")

    data = df[df[target].notna()].reset_index(drop=True)
    if data[target].nunique() < 2:
        raise ValueError("the target needs at least two different values")

    # labels → 0..k-1 in sorted order, so for 'no'/'yes', 0/1 or False/True the "positive" class is 1
    y_cat = pd.Categorical(data[target], categories=sorted(data[target].unique(), key=str), ordered=True)
    y_all = np.asarray(y_cat.codes)
    positive_label = str(y_cat.categories[-1])

    sampled = False
    if len(data) > max_rows:
        keep, _ = train_test_split(np.arange(len(data)), train_size=max_rows, stratify=y_all,
                                   random_state=random_state)
        data, y_all = data.iloc[np.sort(keep)].reset_index(drop=True), y_all[np.sort(keep)]
        sampled = True

    min_class = int(np.bincount(y_all).min())
    folds = min(n_splits, min_class)
    if folds < 2:
        raise ValueError("every class needs at least 2 rows for cross-validation")

    feature_cols = [c for c in data.columns if c not in {target, time_col, *id_cols}]
    if not feature_cols:
        raise ValueError("no feature columns left to check")
    X, categorical = checks.encode_features(data[feature_cols])

    findings: list[checks.Finding] = []

    # 1. each column on its own (plus declared ID columns, which are not modelled)
    feature_auc, found = checks.single_feature_power(X, y_all, categorical, folds, random_state)
    findings += found
    if id_cols:
        X_ids, cat_ids = checks.encode_features(data[id_cols])
        id_auc, _ = checks.single_feature_power(X_ids, y_all, cat_ids, folds, random_state)
        feature_auc.update(id_auc)

    # 2. the full model, then without each of the strongest / suspiciously named columns
    baseline = checks.cv_auc(X, y_all, categorical, folds, random_state)
    hinted = [f.column for f in checks.name_hints(feature_cols)]
    strongest = [c for c, _ in sorted(((c, feature_auc[c]) for c in feature_cols), key=lambda kv: -kv[1])][:top_k]
    candidates = list(dict.fromkeys(strongest + hinted))
    drop_auc, found = checks.dominant_feature(X, y_all, categorical, candidates, baseline, folds, random_state)
    findings += found

    # 3. structure of the data
    findings += checks.id_like(data[feature_cols + id_cols], feature_auc)
    findings += checks.name_hints(feature_cols)
    findings += checks.duplicates(data[feature_cols], pd.Series(y_all))
    if test_df is not None:
        findings += checks.train_test_overlap(df[feature_cols], test_df)

    # 4. time
    if time_col is not None:
        t = data[time_col]
        if not (pd.api.types.is_numeric_dtype(t) or pd.api.types.is_datetime64_any_dtype(t)):
            t = pd.to_datetime(t, errors="coerce")
        order = checks.encode_features(t.to_frame())[0].iloc[:, 0].fillna(np.inf).to_numpy()
        findings += checks.temporal(X, y_all, categorical, order, baseline, random_state, positive_label)

    return Report(
        findings=findings,
        baseline_auc=baseline,
        feature_auc=feature_auc,
        n_rows=len(data),
        n_features=len(feature_cols),
        n_splits=folds,
        sampled=sampled,
        drop_column_auc=drop_auc,
    )
