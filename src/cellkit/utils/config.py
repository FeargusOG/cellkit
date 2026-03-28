"""Utilities for creating experiment and run output directories."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

EXP_DIR = "experiment_dir"
RUNS_DIR = "runs_dir"
RUN_DIR = "run_dir"
LOG_DIR = "logs_dir"
CONFIG_PATH = "config_path"
RUN_CONFIG_PATH = "run_config_path"
FLAT_TRAIN_CONFIG_PATH = "flat_train_config_path"
MANIFEST_PATH = "manifest_path"


def config_for_hash(config: dict[str, Any]) -> dict[str, Any]:
    """Return the subset of config used to identify an experiment.

    ``output_dir`` and ``run_title`` control where the run is written, so they are
    excluded from the stable experiment hash.
    """
    return {
        key: value
        for key, value in config.items()
        if key not in {"output_dir", "run_title"}
    }


def compute_short_sha(config: dict[str, Any], length: int = 7) -> str:
    """Compute a short stable hash for the experiment config."""
    cfg_str = json.dumps(config_for_hash(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(cfg_str.encode("utf-8")).hexdigest()[:length]


def build_experiment_dir(output_dir: Path, short_sha: str) -> Path:
    """Return the experiment directory path for one config."""
    return output_dir / short_sha


def build_run_dir_name(timestamp: str, run_title: str) -> str:
    """Return the run directory name for one timestamped run."""
    return f"{timestamp}_{run_title}"


def write_config_json(config: dict[str, Any], config_path: Path) -> None:
    """Write the canonical config JSON for an experiment."""
    config_path.write_text(json.dumps(config, indent=4, sort_keys=True) + "\n")


def write_run_artifacts(
    *,
    run_dir: Path,
    run_config: dict[str, Any],
    flat_train_config: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Path]:
    """Write per-run plain JSON artifacts for inspection and reproducibility."""
    run_config_path = run_dir / "run_config.json"
    flat_train_config_path = run_dir / "flat_train_config.json"
    manifest_path = run_dir / "manifest.json"

    write_config_json(run_config, run_config_path)
    write_config_json(flat_train_config, flat_train_config_path)
    write_config_json(manifest, manifest_path)

    return {
        RUN_CONFIG_PATH: run_config_path,
        FLAT_TRAIN_CONFIG_PATH: flat_train_config_path,
        MANIFEST_PATH: manifest_path,
    }


def sha1_file(path: str | Path) -> str:
    """Return the SHA1 digest of a file's contents."""
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_commit(repo_dir: str | Path) -> str | None:
    """Return the current git commit SHA for ``repo_dir``, if available."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


def setup_run_dirs(
    train_config_json: dict[str, Any], output_dir: str, run_title: str
) -> dict[str, Path]:
    """Create experiment and run directory structure.

    Structure:
        <output_dir>/
            <short_sha>/
                train_config.json
                runs/
                    <timestamp>_<run_title>/

    Args:
        config: Run configuration dictionary. Must contain ``output_dir`` and
            ``run_title``.

    Returns:
        Dictionary containing created paths and the computed short hash.
    """
    short_sha = compute_short_sha(train_config_json)

    experiment_dir = build_experiment_dir(Path(output_dir), short_sha)
    runs_dir = experiment_dir / "runs"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = runs_dir / build_run_dir_name(timestamp, run_title)

    run_dir.mkdir(parents=True, exist_ok=False)

    train_config_path = experiment_dir / "train_config.json"
    write_config_json(train_config_json, train_config_path)

    return {
        EXP_DIR: experiment_dir,
        RUNS_DIR: runs_dir,
        RUN_DIR: run_dir,
        LOG_DIR: run_dir / "logs",
        CONFIG_PATH: train_config_path,
    }
