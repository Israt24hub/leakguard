import json

from leakguard.cli import main
from leakguard.demo import bank_like


def write_bank_csv(tmp_path, sep=";"):
    path = tmp_path / "bank.csv"
    bank_like(n=1500, seed=2).to_csv(path, sep=sep, index=False)
    return str(path)


def test_exit_code_1_when_high_finding(tmp_path, capsys):
    path = write_bank_csv(tmp_path)
    assert main([path, "--target", "y", "--time-col", "contact_date", "--id-cols", "customer_id", "--top-k", "2"]) == 1
    assert "duration" in capsys.readouterr().out


def test_fail_on_none_always_passes(tmp_path):
    path = write_bank_csv(tmp_path, sep=",")
    assert main([path, "--target", "y", "--top-k", "1", "--fail-on", "none"]) == 0


def test_json_output_with_time_and_id_columns(tmp_path, capsys):
    path = write_bank_csv(tmp_path)
    main([path, "--target", "y", "--time-col", "contact_date", "--id-cols", "customer_id",
          "--top-k", "1", "--format", "json", "--fail-on", "none"])
    data = json.loads(capsys.readouterr().out)
    assert data["n_features"] == 9
    assert any(f["check"] == "label_shift" for f in data["findings"])


def test_bad_target_exits_2(tmp_path, capsys):
    path = write_bank_csv(tmp_path)
    assert main([path, "--target", "missing"]) == 2
    assert "error" in capsys.readouterr().err
