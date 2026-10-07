import numpy as np
import pandas as pd
import pytest

from leakguard import scan
from leakguard.checks import _tokens, name_hints
from leakguard.demo import bank_like


def severities(report, check, column=None):
    return {f.severity for f in report.findings if f.check == check and (column is None or f.column == column)}


@pytest.fixture(scope="module")
def bank():
    return bank_like(n=3000, seed=1)


def test_flags_the_duration_leak(bank):
    report = scan(bank, "y", time_col="contact_date", id_cols=["customer_id"], top_k=3)
    assert "high" in severities(report, "dominant_feature", "duration")
    assert report.drop_column_auc["duration"] < report.baseline_auc - 0.10


def test_duration_still_flagged_when_date_is_a_feature(bank):
    # the date carries real signal too, so the gap shrinks, but duration must still stand out
    report = scan(bank, "y", id_cols=["customer_id"], top_k=3)
    assert severities(report, "dominant_feature", "duration") & {"medium", "high"}


def test_clean_data_has_no_high_findings(bank):
    clean = bank.drop(columns=["duration", "customer_id", "contact_date"])
    report = scan(clean, "y", top_k=3)
    assert not report.has_issues("high")


def test_column_that_is_the_answer_in_disguise():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 800)
    df = pd.DataFrame({"x": rng.normal(size=800), "refund_issued": y, "y": y})
    report = scan(df, "y", top_k=2)
    assert "high" in severities(report, "single_feature", "refund_issued")


def test_id_that_encodes_the_outcome():
    rng = np.random.default_rng(0)
    y = np.sort(rng.integers(0, 2, 600))  # rows exported sorted by outcome
    df = pd.DataFrame({"row_id": np.arange(600), "x": rng.normal(size=600), "y": y})
    report = scan(df, "y", top_k=2)
    assert "medium" in severities(report, "id_like", "row_id")


def test_duplicates_and_train_test_overlap():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.integers(0, 50, 500), "b": rng.normal(size=500).round(2)})
    df["y"] = (df.a > 25).astype(int)
    df = pd.concat([df, df.iloc[:60]], ignore_index=True)  # ~11% duplicated
    report = scan(df, "y", test_df=df.iloc[100:200], top_k=1)
    assert severities(report, "duplicates") & {"medium", "low"}
    assert "high" in severities(report, "train_test_overlap")


def test_temporal_drift_is_reported():
    rng = np.random.default_rng(0)
    n = 3000
    t = pd.date_range("2020-01-01", periods=n, freq="h")
    x = rng.normal(size=n)
    later = np.arange(n) >= int(n * 0.8)
    # the relationship flips in the last fifth of the data
    p = 1 / (1 + np.exp(-np.where(later, -3 * x, 3 * x)))
    df = pd.DataFrame({"when": t, "x": x, "noise": rng.normal(size=n), "y": rng.random(n) < p})
    report = scan(df, "y", time_col="when", top_k=1)
    assert "high" in severities(report, "temporal_drop")


def test_multiclass_missing_values_and_mixed_types():
    rng = np.random.default_rng(0)
    n = 900
    df = pd.DataFrame({
        "num": np.where(rng.random(n) < 0.1, np.nan, rng.normal(size=n)),
        "text": pd.array(rng.choice(["a", "b", None], n), dtype="string"),
        "flag": rng.random(n) < 0.5,
        "seen": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 100, n), unit="D"),
        "y": rng.choice(["cat", "dog", "bird"], n),
    })
    report = scan(df, "y", top_k=2)
    assert 0.3 < report.baseline_auc < 0.7
    assert set(report.feature_auc) == {"num", "text", "flag", "seen"}


def test_large_data_is_sampled():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=3000)})
    df["y"] = df.x + rng.normal(size=3000) > 0
    report = scan(df, "y", max_rows=1000, top_k=1)
    assert report.sampled and report.n_rows == 1000


@pytest.mark.parametrize("kwargs, message", [
    ({"target": "nope"}, "target column"),
    ({"target": "y", "time_col": "nope"}, "not in the dataframe"),
])
def test_bad_arguments(kwargs, message):
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [0, 1, 0, 1]})
    with pytest.raises(ValueError, match=message):
        scan(df, **kwargs)


def test_single_class_target_is_rejected():
    df = pd.DataFrame({"x": range(10), "y": [1] * 10})
    with pytest.raises(ValueError, match="two different values"):
        scan(df, "y")


def test_name_tokens_and_hints():
    assert _tokens("callDuration_sec") == ["call", "duration", "sec"]
    assert [f.column for f in name_hints(["age", "final_status", "isPaid"])] == ["final_status", "isPaid"]


def test_report_outputs(bank):
    report = scan(bank, "y", time_col="contact_date", top_k=1)
    frame = report.to_frame()
    assert list(frame.columns) == ["check", "column", "severity", "score", "detail"]
    assert "| Severity |" in report.to_markdown()
    assert report.worst == "high"
    assert "LeakGuard report" in str(report)
