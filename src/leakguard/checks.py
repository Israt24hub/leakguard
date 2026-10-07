"""The individual leakage checks.

Every check returns a list of :class:`Finding`. A finding is a *suspect*, not a verdict:
LeakGuard points at columns and rows worth a second look, and a person decides whether
the information would really be available at prediction time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

SEVERITIES = ("high", "medium", "low", "info")
SEVERITY_RANK = {"high": 3, "medium": 2, "low": 1, "info": 0}

# words that often name information recorded *after* the outcome
_HINT_WORDS = {
    "duration", "outcome", "result", "results", "status", "response", "final", "after",
    "post", "closed", "closing", "churn", "churned", "paid", "approved", "approval",
    "label", "target", "decision", "refund", "refunded", "cancel", "cancelled", "canceled",
    "resolved", "resolution", "converted", "conversion", "success", "successful", "won", "lost",
}
_ID_WORDS = {"id", "uuid", "guid", "index", "idx", "key", "serial", "row", "rowid", "pk"}


@dataclass(frozen=True)
class Finding:
    """One suspicious thing LeakGuard found."""

    check: str
    column: str | None
    severity: str
    score: float | None
    detail: str


# --------------------------------------------------------------------------- helpers


def _tokens(name: str) -> list[str]:
    """Split a column name like 'callDuration_sec' into ['call', 'duration', 'sec']."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    return [t for t in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if t]


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[bool]]:
    """Turn any dataframe into floats a tree model can use.

    Numbers stay numbers, dates become seconds since 1970, and everything else (text,
    categories) becomes category codes. Missing values stay missing (NaN).
    Returns the encoded frame and a mask saying which columns are categorical.
    """
    columns: dict[str, pd.Series] = {}
    categorical: list[bool] = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            columns[col] = s.astype("float64")
            categorical.append(False)
        elif pd.api.types.is_numeric_dtype(s):
            columns[col] = pd.to_numeric(s, errors="coerce").astype("float64")
            categorical.append(False)
        elif pd.api.types.is_datetime64_any_dtype(s):
            columns[col] = s.map(lambda v: v.timestamp() if pd.notna(v) else np.nan).astype("float64")
            categorical.append(False)
        else:
            codes = s.astype("category").cat.codes.astype("float64").replace(-1.0, np.nan)
            columns[col] = codes
            # the boosting model handles up to ~250 categories natively; beyond that, use codes as numbers
            categorical.append(bool(codes.nunique() <= 250))
    return pd.DataFrame(columns, index=df.index), categorical


def _model(categorical: list[bool], seed: int) -> HistGradientBoostingClassifier:
    mask = np.array(categorical, dtype=bool)
    return HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        max_leaf_nodes=15,
        categorical_features=mask if mask.any() else None,
        random_state=seed,
    )


def auc(y: np.ndarray, proba: np.ndarray) -> float:
    """ROC-AUC for binary or multiclass (one-vs-rest, macro) targets."""
    if proba.shape[1] == 2:
        return float(roc_auc_score(y, proba[:, 1]))
    return float(roc_auc_score(y, proba, multi_class="ovr", average="macro"))


def cv_auc(X: pd.DataFrame, y: np.ndarray, categorical: list[bool], n_splits: int, seed: int) -> float:
    """Out-of-fold ROC-AUC of a gradient-boosting model on X."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba = cross_val_predict(_model(categorical, seed), X, y, cv=cv, method="predict_proba")
    return auc(y, proba)


def _fmt(x: float) -> str:
    return f"{x:.3f}"


# --------------------------------------------------------------------------- checks


def single_feature_power(
    X: pd.DataFrame, y: np.ndarray, categorical: list[bool], n_splits: int, seed: int
) -> tuple[dict[str, float], list[Finding]]:
    """How well does each column predict the target *on its own*?

    A column that alone gets ROC-AUC ≈ 1.0 is almost always the answer in disguise
    (e.g. 'refund_issued' when predicting 'order_cancelled').
    """
    scores: dict[str, float] = {}
    findings: list[Finding] = []
    for i, col in enumerate(X.columns):
        if X[col].nunique(dropna=True) < 2:
            scores[col] = 0.5
            continue
        score = cv_auc(X[[col]], y, [categorical[i]], n_splits, seed)
        scores[col] = score
        if score >= 0.98:
            findings.append(Finding("single_feature", col, "high", score,
                f"On its own this column predicts the target almost perfectly (ROC-AUC {_fmt(score)}). "
                "It is very likely the answer in disguise."))
        elif score >= 0.90:
            findings.append(Finding("single_feature", col, "medium", score,
                f"On its own this column predicts the target very well (ROC-AUC {_fmt(score)}). "
                "Check that its value exists before the outcome happens."))
    return scores, findings


def dominant_feature(
    X: pd.DataFrame, y: np.ndarray, categorical: list[bool], candidates: list[str],
    baseline: float, n_splits: int, seed: int,
) -> tuple[dict[str, float], list[Finding]]:
    """Does the model fall apart without one column? (drop-column test)

    This is how the `duration` leak in the UCI Bank Marketing data shows up: call
    duration is only known after the call, yet removing it costs ~0.14 ROC-AUC.
    """
    without: dict[str, float] = {}
    findings: list[Finding] = []
    if X.shape[1] < 2:
        return without, findings
    for col in candidates:
        keep = [c for c in X.columns if c != col]
        mask = [categorical[list(X.columns).index(c)] for c in keep]
        score = cv_auc(X[keep], y, mask, n_splits, seed)
        without[col] = score
        gap = baseline - score
        detail = (f"Removing it drops ROC-AUC from {_fmt(baseline)} to {_fmt(score)} (−{gap:.3f}). "
                  "The model leans heavily on this one column: make sure its value is known at the "
                  "moment a prediction would be made.")
        if gap >= 0.10:
            findings.append(Finding("dominant_feature", col, "high", gap, detail))
        elif gap >= 0.05:
            findings.append(Finding("dominant_feature", col, "medium", gap, detail))
    return without, findings


def id_like(df: pd.DataFrame, feature_auc: dict[str, float]) -> list[Finding]:
    """Identifier columns: unique per row, or named like an ID.

    An ID that predicts the target usually means the rows were sorted or numbered by
    outcome, so the model can learn the numbering instead of the problem.
    """
    findings: list[Finding] = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        named = bool(set(_tokens(col)) & _ID_WORDS)
        unique_ratio = s.nunique(dropna=True) / max(n, 1)
        is_float = pd.api.types.is_float_dtype(s)
        looks_unique = unique_ratio > 0.98 and not is_float and n >= 20
        if not (named or looks_unique):
            continue
        score = feature_auc.get(col, 0.5)
        if score >= 0.60:
            findings.append(Finding("id_like", col, "medium", score,
                f"Looks like an identifier, yet it predicts the target (ROC-AUC {_fmt(score)}). "
                "Rows were probably ordered or numbered by outcome. Drop it before training."))
        else:
            findings.append(Finding("id_like", col, "low", score,
                "Looks like an identifier. It carries no real signal; drop it before training."))
    return findings


def name_hints(columns: list[str]) -> list[Finding]:
    """Column names that suggest information recorded after the outcome."""
    findings: list[Finding] = []
    for col in columns:
        hits = sorted(set(_tokens(col)) & _HINT_WORDS)
        if hits:
            findings.append(Finding("name_hint", col, "low", None,
                f"The name contains '{hits[0]}', which often means the value is recorded after "
                "the outcome. Confirm when this value becomes available."))
    return findings


def duplicates(df: pd.DataFrame, y: pd.Series) -> list[Finding]:
    """Repeated rows: with a random split, copies land in both train and test."""
    if df.empty:
        return []
    keys = pd.util.hash_pandas_object(df.astype("string"), index=False)
    dup_mask = keys.duplicated(keep=False)
    frac = float(dup_mask.mean())
    if frac <= 0.01:
        return []
    conflicting = (pd.DataFrame({"k": keys[dup_mask], "y": y[dup_mask]}).groupby("k")["y"].nunique() > 1).sum()
    severity = "medium" if frac > 0.05 else "low"
    detail = (f"{frac:.1%} of rows have an exact copy. A random train/test split puts copies on both "
              "sides, so the model is partly tested on rows it has already seen.")
    if conflicting:
        detail += f" {conflicting} duplicated row groups even have different labels."
    return [Finding("duplicates", None, severity, frac, detail)]


def train_test_overlap(train: pd.DataFrame, test: pd.DataFrame) -> list[Finding]:
    """Rows of the test set that also appear in the training set."""
    common = [c for c in train.columns if c in test.columns]
    if not common or test.empty:
        return []
    train_keys = set(pd.util.hash_pandas_object(train[common].astype("string"), index=False))
    test_keys = pd.util.hash_pandas_object(test[common].astype("string"), index=False)
    overlap = float(test_keys.isin(train_keys).mean())
    if overlap == 0:
        return []
    severity = "high" if overlap > 0.01 else "medium"
    return [Finding("train_test_overlap", None, severity, overlap,
        f"{overlap:.1%} of test rows also appear in the training set, so test scores are inflated.")]


def temporal(
    X: pd.DataFrame, y: np.ndarray, categorical: list[bool], order: np.ndarray,
    random_auc: float, seed: int, positive_label: str = "1",
) -> list[Finding]:
    """Train on the past, test on the future, and compare with a random split.

    In the Bank Marketing data, training on 2008–2009 and testing on 2010 drops ROC-AUC to
    about 0.65, and the share of 'yes' answers jumps from ~11% to ~52%.
    """
    findings: list[Finding] = []
    idx = np.argsort(order, kind="stable")
    cut = int(len(idx) * 0.8)
    train_idx, test_idx = idx[:cut], idx[cut:]
    y_train, y_test = y[train_idx], y[test_idx]

    if len(np.unique(y_train)) == len(np.unique(y)) and len(np.unique(y_test)) == len(np.unique(y)):
        model = _model(categorical, seed).fit(X.iloc[train_idx], y_train)
        time_auc = auc(y_test, model.predict_proba(X.iloc[test_idx]))
        gap = random_auc - time_auc
        detail = (f"Random-split ROC-AUC is {_fmt(random_auc)}, but training on the earliest 80% and "
                  f"testing on the latest 20% gives {_fmt(time_auc)}. Random splits overestimate how "
                  "the model will do on future data.")
        if gap >= 0.10:
            findings.append(Finding("temporal_drop", None, "high", gap, detail))
        elif gap >= 0.05:
            findings.append(Finding("temporal_drop", None, "medium", gap, detail))
    else:
        findings.append(Finding("temporal_drop", None, "info", None,
            "Some classes never appear in the earliest 80% or the latest 20% of rows, so a "
            "time-based test was not possible."))

    if len(np.unique(y)) == 2:
        p_train, p_test = float(y_train.mean()), float(y_test.mean())
        if abs(p_test - p_train) >= 0.15 or (min(p_train, p_test) > 0 and max(p_train, p_test) / min(p_train, p_test) >= 2):
            findings.append(Finding("label_shift", None, "medium", p_test - p_train,
                f"The share of '{positive_label}' labels changes from {p_train:.0%} (earliest 80%) to "
                f"{p_test:.0%} (latest 20%). Compare scores against a simple baseline on the same "
                "period instead of comparing raw F1 across periods."))
    return findings
