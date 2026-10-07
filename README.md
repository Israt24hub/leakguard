# LeakGuard

[![tests](https://github.com/Israt24hub/leakguard/actions/workflows/tests.yml/badge.svg)](https://github.com/Israt24hub/leakguard/actions/workflows/tests.yml)

**Find data leakage before it fools your model.**

Data leakage happens when a model is trained on information it would not have when making a real prediction. The scores look great; the model fails in practice.

## Why I built this

In our CSE445 project on the UCI Bank Marketing dataset ([write-up](https://github.com/Israt24hub/bank-marketing-leakage)), one column, `duration` (the length of the sales call), raised ROC-AUC by about **0.14** in every model we tried: classical ML, tabular foundation models, fine-tuned transformers and LLM prompting. The problem: call length is only known *after* the call, when the customer has already said yes or no. The leak was bigger than any difference between the models.

Finding it took manual work. LeakGuard runs the same checks automatically on any table.

## Install

```bash
pip install git+https://github.com/Israt24hub/leakguard
```

## Quick start

```python
import pandas as pd
from leakguard import scan

df = pd.read_csv("bank-full.csv", sep=";")
report = scan(df, target="y")
print(report)
```

Or from the command line:

```bash
leakguard bank.csv --target y --time-col contact_date --id-cols customer_id
```

## Example

`examples/demo.py` runs LeakGuard on a synthetic copy of the Bank Marketing data (`leakguard.demo.bank_like()`, built with the same traps):

```text
LeakGuard report
6,000 rows, 9 features, baseline ROC-AUC 0.845 (5-fold CV)

HIGH    dominant_feature    duration
        Removing it drops ROC-AUC from 0.845 to 0.677 (−0.168). The model leans heavily on this one column: make sure its value is known at the moment a prediction would be made.
MEDIUM  label_shift         (dataset)
        The share of 'yes' labels changes from 7% (earliest 80%) to 26% (latest 20%). Compare scores against a simple baseline on the same period instead of comparing raw F1 across periods.
LOW     id_like             customer_id
        Looks like an identifier. It carries no real signal; drop it before training.
LOW     name_hint           duration
        The name contains 'duration', which often means the value is recorded after the outcome. Confirm when this value becomes available.

Strongest single columns (ROC-AUC on their own):
  duration                 0.797
  poutcome                 0.594
  previous                 0.577
  campaign                 0.562
  job                      0.547
```

## What it checks

| Check | Catches | How |
|---|---|---|
| `single_feature` | A column that *is* the answer in disguise (e.g. `refund_issued` when predicting cancellations) | Cross-validated ROC-AUC of each column on its own; ≥ 0.90 is suspicious, ≥ 0.98 almost certain |
| `dominant_feature` | A column the whole model depends on, like `duration` | Drop-column test on the strongest and suspiciously named columns: how much ROC-AUC falls without it |
| `id_like` | Row IDs, and IDs that predict the target because the data was sorted by outcome | Uniqueness, column name, and the ID's own predictive power |
| `name_hint` | Columns named like after-the-fact information (`status`, `final_`, `duration`, `paid`…) | Name tokens, including camelCase |
| `duplicates` | Repeated rows that end up in both train and test after a random split | Row hashes, plus duplicates with conflicting labels |
| `train_test_overlap` | Test rows that also appear in training (pass `test_df=`) | Row hashes across the two sets |
| `temporal_drop` | Models that only look good because the test set is mixed in time with the training set (pass `time_col=`) | Trains on the earliest 80% and tests on the latest 20%, then compares with random-split ROC-AUC |
| `label_shift` | Big changes in the share of each label over time, which make raw F1 comparisons across periods misleading | Positive rate in the earliest 80% vs the latest 20% |

The model behind the checks is scikit-learn's `HistGradientBoostingClassifier`, which handles missing values and categories directly. Text, categories, booleans and dates are encoded automatically.

## Reading the report

A finding is a **suspect, not a verdict**. A strong column can be perfectly legitimate. The question LeakGuard makes you answer is: *at the moment I would make this prediction, would I already know this value?*

- **high**: almost certainly a problem, or a big effect on the scores. Investigate before trusting any result.
- **medium**: worth checking.
- **low**: housekeeping (identifiers, suspicious names).
- **info**: a check could not run (for example, a class missing from one time period).

## Use it in CI

The command exits with code **1** if any finding is at least as severe as `--fail-on` (default `high`), **0** otherwise, and **2** on bad input. That makes it easy to block a training pipeline when someone adds a leaky column:

```bash
leakguard data/train.csv --target churned --time-col signup_date --fail-on high
```

Other options: `--test test.csv` (overlap check), `--format markdown|json`, `--top-k 5` (columns given the slower drop-column test), `--max-rows 20000` (larger files are sampled).

## Limitations

- Classification targets only (binary and multiclass). Regression is not supported yet.
- The drop-column test can miss a leak spread across several correlated columns, because the others fill in when one is removed.
- LeakGuard cannot know *when* a value is recorded in your business; it can only point at columns that behave like leaks.
- Datasets larger than `max_rows` are sampled, so scores vary slightly between runs with different seeds.

## Development

```bash
pip install -e ".[dev]"
pytest
```

Tests run on every push for Python 3.10 to 3.13 (see `.github/workflows/tests.yml`).

## License

MIT © Israt Zaman Srity
