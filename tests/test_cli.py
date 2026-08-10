def test_run_with_a_missing_config_reports_one_line_without_a_traceback(tmp_path, capsys):
    import cli

    missing = tmp_path / "no-existe.yaml"
    exit_code = cli.main(["run", str(missing), "--runs-dir", str(tmp_path / "runs")])

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Traceback" not in captured.err
    lines = [line for line in captured.err.splitlines() if line.strip()]
    assert len(lines) == 1
    assert "Error" in lines[0]
