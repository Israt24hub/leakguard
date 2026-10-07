"""The report returned by :func:`leakguard.scan`."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import pandas as pd

from .checks import SEVERITY_RANK, Finding


@dataclass
class Report:
    findings: list[Finding]
    baseline_auc: float
    feature_auc: dict[str, float]
    n_rows: int
    n_features: int
    n_splits: int
    sampled: bool = False
    drop_column_auc: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.findings = sorted(
            self.findings, key=lambda f: (-SEVERITY_RANK[f.severity], f.check, str(f.column))
        )

    # ------------------------------------------------------------------ queries

    @property
    def worst(self) -> str | None:
        """The most serious severity found, or None if nothing was found."""
        return self.findings[0].severity if self.findings else None

    def has_issues(self, min_severity: str = "medium") -> bool:
        """True if any finding is at least `min_severity` ("high", "medium", "low" or "info")."""
        return any(SEVERITY_RANK[f.severity] >= SEVERITY_RANK[min_severity] for f in self.findings)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(f) for f in self.findings],
                            columns=["check", "column", "severity", "score", "detail"])

    # ------------------------------------------------------------------ output

    def _header(self) -> str:
        sample = " (sampled)" if self.sampled else ""
        return (f"{self.n_rows:,} rows{sample}, {self.n_features} features, "
                f"baseline ROC-AUC {self.baseline_auc:.3f} ({self.n_splits}-fold CV)")

    def __str__(self) -> str:
        lines = ["LeakGuard report", self._header(), ""]
        if not self.findings:
            lines.append("No leakage suspects found.")
        for f in self.findings:
            where = f.column if f.column is not None else "(dataset)"
            lines.append(f"{f.severity.upper():<7} {f.check:<19} {where}")
            lines.append(f"        {f.detail}")
        top = sorted(self.feature_auc.items(), key=lambda kv: -kv[1])[:5]
        if top:
            lines += ["", "Strongest single columns (ROC-AUC on their own):"]
            lines += [f"  {name:<24} {score:.3f}" for name, score in top]
        return "\n".join(lines)

    def to_markdown(self) -> str:
        out = [f"**LeakGuard report**: {self._header()}", ""]
        if not self.findings:
            return "\n".join(out + ["No leakage suspects found."])
        out += ["| Severity | Check | Column | Detail |", "|---|---|---|---|"]
        for f in self.findings:
            col = f"`{f.column}`" if f.column is not None else "*(dataset)*"
            out.append(f"| {f.severity} | {f.check} | {col} | {f.detail} |")
        return "\n".join(out)

    def to_json(self) -> str:
        return json.dumps({
            "n_rows": self.n_rows,
            "n_features": self.n_features,
            "baseline_auc": self.baseline_auc,
            "feature_auc": self.feature_auc,
            "findings": [asdict(f) for f in self.findings],
        }, indent=2)
