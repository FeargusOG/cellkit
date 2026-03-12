"""Utilities for creating experiment and run output directories."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def validate_run_config(config: dict[str, Any]) -> tuple[Path, str]:
    """Validate required run config fields.

    Args:
        config: Run configuration dictionary.

    Returns:
        Tuple of ``(output_dir, run_title)``.

    Raises:
        ValueError: If required fields are missing or empty.
    """
    if "output_dir" not in config:
        raise ValueError("config must contain 'output_dir'")
    if "run_title" not in config:
        raise ValueError("config must contain 'run_title'")

    output_dir_value = str(config["output_dir"]).strip()
    run_title = str(config["run_title"]).strip()

    if not output_dir_value:
        raise ValueError("'output_dir' must not be empty")
    if not run_title:
        raise ValueError("'run_title' must not be empty")

    output_dir = Path(output_dir_value)
    return output_dir, run_title


def config_for_hash(config: dict[str, Any]) -> dict[str, Any]:
    """Return the subset of config used to identify an experiment.

    ``output_dir`` and ``run_title`` control where the run is written, so they are
    excluded from the stable experiment hash.
    """
    return {
        key: value for key, value in config.items() if key not in {"output_dir", "run_title"}
    }


def compute_short_sha(config: dict[str, Any], length: int = 7) -> str:
    """Compute a short stable hash for the experiment config."""
    cfg_str = json.dumps(config_for_hash(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(cfg_str.encode("utf-8")).hexdigest()[:length]


def build_experiment_dir(output_dir: Path, short_sha: str, run_title: str) -> Path:
    """Return the experiment directory path for one config."""
    return output_dir / f"{short_sha}_{run_title}"


def write_config_json(config: dict[str, Any], config_path: Path) -> None:
    """Write the canonical config JSON for an experiment."""
    config_path.write_text(json.dumps(config, indent=4, sort_keys=True) + "\n")


def setup_run_dirs(
    config: dict[str, Any]
) -> dict[str, Path | str]:
    """Create experiment and run directory structure.

    Structure:
        <output_dir>/
            <short_sha>_<run_title>/
                config.json
                runs/
                    <timestamp>/

    Args:
        config: Run configuration dictionary. Must contain ``output_dir`` and
            ``run_title``.

    Returns:
        Dictionary containing created paths and the computed short hash.
    """
    output_dir, run_title = validate_run_config(config)
    short_sha = compute_short_sha(config)

    experiment_dir = build_experiment_dir(output_dir, short_sha, run_title)
    runs_dir = experiment_dir / "runs"
    run_dir = runs_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    run_dir.mkdir(parents=True, exist_ok=False)

    config_path = experiment_dir / "config.json"
    write_config_json(config, config_path)

    return {
        "experiment_dir": experiment_dir,
        "run_dir": run_dir,
        "runs_dir": runs_dir,
        "config_path": config_path,
        "short_sha": short_sha,
    }
