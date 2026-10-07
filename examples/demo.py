"""Run LeakGuard on a synthetic copy of the UCI Bank Marketing data.

    python examples/demo.py
"""

from leakguard import scan
from leakguard.demo import bank_like

df = bank_like()  # 6,000 phone calls; 'duration' is only known after the call ends
report = scan(df, target="y", time_col="contact_date", id_cols=["customer_id"])
print(report)
