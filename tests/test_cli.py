# tests/test_cli.py
import numpy as np
import pytest

from csm_processor._version import __version__
from csm_processor.cli import main


def test_cli_version_outputs_once(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])

    out = capsys.readouterr().out.strip()
    assert out.endswith(f" {__version__}")
    assert out.count(__version__) == 1


def test_cli_missing_file_fails_cleanly():
    with pytest.raises(SystemExit) as excinfo:
        main(["does_not_exist.csv", "--fs", "1000", "--nrec", "256"])

    assert "Input file(s) not found:" in str(excinfo.value)


def test_cli_processes_csv_and_writes_output(tmp_path, capsys):
    t = np.arange(1024) / 1000.0
    data = np.column_stack([
        np.sin(2 * np.pi * 50 * t),
        np.sin(2 * np.pi * 50 * t + 0.3),
    ])

    csv_path = tmp_path / "demo.csv"
    np.savetxt(csv_path, data, delimiter=",")

    main([
        str(csv_path),
        "--fs", "1000",
        "--nrec", "256",
        "--outdir", str(tmp_path),
    ])

    out = capsys.readouterr().out
    assert "✗ Error" not in out
    assert (tmp_path / "demo_CSM.npz").exists()