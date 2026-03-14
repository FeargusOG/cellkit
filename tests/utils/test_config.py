import json
from datetime import datetime
from pathlib import Path
from unittest import mock

import pytest

from cellkit.utils.config import build_experiment_dir
from cellkit.utils.config import build_run_dir_name
from cellkit.utils.config import compute_short_sha
from cellkit.utils.config import config_for_hash
from cellkit.utils.config import setup_run_dirs
from cellkit.utils.config import write_config_json


def test_config_for_hash_excludes_output_dir_and_run_title():
    config = {
        "output_dir": "runs",
        "run_title": "demo",
        "seed": 0,
        "batch_size": 32,
    }

    assert config_for_hash(config) == {"seed": 0, "batch_size": 32}


def test_compute_short_sha_ignores_output_location_metadata():
    config_a = {
        "output_dir": "runs_a",
        "run_title": "demo-a",
        "seed": 0,
        "batch_size": 32,
    }
    config_b = {
        "output_dir": "runs_b",
        "run_title": "demo-b",
        "seed": 0,
        "batch_size": 32,
    }

    assert compute_short_sha(config_a) == compute_short_sha(config_b)


def test_build_experiment_dir_uses_short_sha_only():
    experiment_dir = build_experiment_dir(Path("runs"), "abc1234")
    assert experiment_dir == Path("runs/abc1234")


def test_build_run_dir_name_uses_timestamp_and_run_title():
    run_dir_name = build_run_dir_name("2026-03-12_10-22-51", "demo")
    assert run_dir_name == "2026-03-12_10-22-51_demo"


def test_write_config_json_writes_sorted_json_with_newline(tmp_path: Path):
    config = {"b": 2, "a": 1}
    config_path = tmp_path / "config.json"

    write_config_json(config, config_path)

    assert config_path.read_text() == '{\n    "a": 1,\n    "b": 2\n}\n'


def test_setup_run_dirs_creates_expected_structure(tmp_path: Path):
    config = {
        "output_dir": str(tmp_path / "outputs"),
        "run_title": "demo",
        "seed": 0,
        "batch_size": 32,
    }
    now = datetime(2026, 3, 12, 10, 22, 51)

    with mock.patch("cellkit.utils.config.datetime") as mocked_datetime:
        mocked_datetime.now.return_value = now
        mocked_datetime.strftime = datetime.strftime
        result = setup_run_dirs(config, config["output_dir"], config["run_title"])

    expected_short_sha = compute_short_sha(config)
    expected_experiment_dir = tmp_path / "outputs" / expected_short_sha
    expected_run_dir = expected_experiment_dir / "runs" / "2026-03-12_10-22-51_demo"

    assert result["experiment_dir"] == expected_experiment_dir
    assert result["runs_dir"] == expected_experiment_dir / "runs"
    assert result["run_dir"] == expected_run_dir
    assert result["config_path"] == expected_experiment_dir / "train_config.json"
    assert expected_run_dir.is_dir()
    assert json.loads((expected_experiment_dir / "train_config.json").read_text()) == config


def test_setup_run_dirs_raises_if_run_dir_already_exists(tmp_path: Path):
    config = {
        "output_dir": str(tmp_path / "outputs"),
        "run_title": "demo",
        "seed": 0,
    }
    now = datetime(2026, 3, 12, 10, 22, 51)

    with mock.patch("cellkit.utils.config.datetime") as mocked_datetime:
        mocked_datetime.now.return_value = now
        mocked_datetime.strftime = datetime.strftime
        setup_run_dirs(config, config["output_dir"], config["run_title"])

    with mock.patch("cellkit.utils.config.datetime") as mocked_datetime:
        mocked_datetime.now.return_value = now
        mocked_datetime.strftime = datetime.strftime
        with pytest.raises(FileExistsError):
            setup_run_dirs(config, config["output_dir"], config["run_title"])
