"""A synthetic dataset shaped like the UCI Bank Marketing data, with the same traps.

The real data (Moro et al., 2014) records phone calls offering a term deposit. Its
`duration` column is the call length, which is only known after the call has ended,
when the customer has already said yes or no. Here it is generated *from* the answer,
so it leaks it on purpose. Later contacts also have a much higher share of "yes",
like the 2010 part of the real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def bank_like(n: int = 6000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    jobs = np.array(["admin.", "blue-collar", "technician", "services", "management",
                     "retired", "student", "unemployed", "self-employed", "housemaid"])
    job = rng.choice(jobs, n, p=[.2, .22, .16, .1, .14, .05, .03, .03, .04, .03])
    age = np.clip(rng.normal(41, 10, n) + np.where(job == "retired", 22, 0) - np.where(job == "student", 16, 0), 18, 90).round()
    balance = np.round(rng.lognormal(6.5, 1.3, n) - 300)
    contact = rng.choice(["cellular", "telephone", "unknown"], n, p=[.65, .07, .28])
    month = rng.choice(["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], n,
                       p=[.02, .07, .3, .12, .15, .14, .03, .04, .11, .02])
    campaign = 1 + rng.poisson(1.6, n)
    previous = rng.poisson(0.3, n)
    poutcome = np.where(previous == 0, "nonexistent", rng.choice(["failure", "success"], n, p=[.7, .3]))
    # contact dates over three years; the last stretch has far more "yes", like 2010
    date = pd.Timestamp("2008-05-01") + pd.to_timedelta(np.sort(rng.uniform(0, 3 * 365, n)), unit="D")
    late = (date >= pd.Timestamp("2010-05-01")).astype(float)

    # what drives a "yes" also drifts over time: in the last year the month and contact
    # effects change, so a model trained on the past does worse on the future
    logit = (-3.4
             + 1.2 * np.isin(job, ["retired", "student"])
             + 2.6 * (poutcome == "success")
             + (1.0 - 1.4 * late) * np.isin(month, ["mar", "sep", "oct", "dec"])
             + (0.6 * late - 0.5) * (month == "may")
             - 0.3 * (campaign - 1)
             - (1.4 - 1.2 * late) * (contact == "unknown")
             + 0.0001 * np.clip(balance, -2000, 20000)
             + 2.4 * late)
    y = rng.random(n) < 1 / (1 + np.exp(-logit))

    # the leak: call length depends on the answer the call produced
    duration = np.round(np.exp(np.where(y, rng.normal(5.95, 0.55, n), rng.normal(5.25, 0.6, n))))

    return pd.DataFrame({
        "customer_id": rng.permutation(np.arange(100001, 100001 + n)),
        "contact_date": date,
        "age": age.astype(int),
        "job": job,
        "balance": balance.astype(int),
        "contact": contact,
        "month": month,
        "campaign": campaign,
        "previous": previous,
        "poutcome": poutcome,
        "duration": duration.astype(int),
        "y": np.where(y, "yes", "no"),
    })
