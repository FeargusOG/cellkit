from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from accelerate import Accelerator
import torch

CHECKPOINTS_DIRNAME = "checkpoints"
LAST_CHECKPOINT_NAME = "last.pt"
BEST_CHECKPOINT_NAME = "best.pt"
METADATA_NAME = "metadata.json"


def checkpoint_paths(run_dir: Path) -> dict[str, Path]:
    checkpoints_dir = run_dir / CHECKPOINTS_DIRNAME
    return {
        "dir": checkpoints_dir,
        "last": checkpoints_dir / LAST_CHECKPOINT_NAME,
        "best": checkpoints_dir / BEST_CHECKPOINT_NAME,
        "metadata": checkpoints_dir / METADATA_NAME,
    }


def save_checkpoint(
    *,
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    run_dir: Path,
    epoch: int,
    global_step: int,
    train_loss: float,
    eval_loss: float,
    best_eval_loss: float | None,
    train_config: dict[str, Any],
    run_config: dict[str, Any],
) -> float:
    paths = checkpoint_paths(run_dir)
    checkpoint_dir = paths["dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    next_best_eval_loss = eval_loss
    if best_eval_loss is not None:
        next_best_eval_loss = min(best_eval_loss, eval_loss)

    last_record = _build_checkpoint_record(
        filename=LAST_CHECKPOINT_NAME,
        epoch=epoch,
        global_step=global_step,
        train_loss=train_loss,
        eval_loss=eval_loss,
        is_best=False,
    )
    _save_checkpoint_payload(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        path=paths["last"],
        record=last_record,
        best_eval_loss=next_best_eval_loss,
        train_config=train_config,
        run_config=run_config,
    )

    metadata = load_checkpoint_metadata(paths["metadata"])
    metadata["last"] = last_record

    if best_eval_loss is None or eval_loss < best_eval_loss:
        best_record = _build_checkpoint_record(
            filename=BEST_CHECKPOINT_NAME,
            epoch=epoch,
            global_step=global_step,
            train_loss=train_loss,
            eval_loss=eval_loss,
            is_best=True,
        )
        _save_checkpoint_payload(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            path=paths["best"],
            record=best_record,
            best_eval_loss=next_best_eval_loss,
            train_config=train_config,
            run_config=run_config,
        )
        metadata["best"] = best_record

    if accelerator.is_main_process:
        paths["metadata"].write_text(json.dumps(metadata, indent=4, sort_keys=True) + "\n")
    accelerator.wait_for_everyone()

    return next_best_eval_loss


def load_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path)
    if checkpoint.suffix != ".pt":
        raise ValueError("resume_from_checkpoint must point to a .pt checkpoint file")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")
    return torch.load(checkpoint, map_location="cpu")


def load_checkpoint_metadata(metadata_path: Path) -> dict[str, Any]:
    if not metadata_path.exists():
        return {"last": None, "best": None}
    return json.loads(metadata_path.read_text())


def _build_checkpoint_record(
    *,
    filename: str,
    epoch: int,
    global_step: int,
    train_loss: float,
    eval_loss: float,
    is_best: bool,
) -> dict[str, Any]:
    return {
        "path": filename,
        "epoch": epoch,
        "global_step": global_step,
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "is_best": is_best,
    }


def _save_checkpoint_payload(
    *,
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    path: Path,
    record: dict[str, Any],
    best_eval_loss: float,
    train_config: dict[str, Any],
    run_config: dict[str, Any],
) -> None:
    payload = {
        "model_state_dict": accelerator.get_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": record["epoch"],
        "global_step": record["global_step"],
        "train_loss": record["train_loss"],
        "eval_loss": record["eval_loss"],
        "best_eval_loss": best_eval_loss,
        "is_best": record["is_best"],
        "train_config": train_config,
        "run_config": run_config,
    }
    accelerator.save(payload, path)
    accelerator.wait_for_everyone()
