import json
from pathlib import Path

import torch

from cellkit.utils.checkpoints import (
    BEST_CHECKPOINT_NAME,
    LAST_CHECKPOINT_NAME,
    checkpoint_paths,
    load_checkpoint_metadata,
    save_checkpoint,
)


class FakeAccelerator:
    def __init__(self):
        self.is_main_process = True

    def get_state_dict(self, model: torch.nn.Module):
        return model.state_dict()

    def save(self, payload, path: Path):
        torch.save(payload, path)

    def wait_for_everyone(self):
        return None


def test_checkpoint_paths_are_created_under_run_dir(tmp_path):
    paths = checkpoint_paths(tmp_path)

    assert paths["dir"] == tmp_path / "checkpoints"
    assert paths["last"] == tmp_path / "checkpoints" / LAST_CHECKPOINT_NAME
    assert paths["best"] == tmp_path / "checkpoints" / BEST_CHECKPOINT_NAME
    assert paths["metadata"] == tmp_path / "checkpoints" / "metadata.json"


def test_save_checkpoint_writes_last_best_and_metadata(tmp_path):
    accelerator = FakeAccelerator()
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)

    best_eval_loss = save_checkpoint(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        run_dir=tmp_path,
        epoch=1,
        global_step=5,
        train_loss=0.8,
        eval_loss=0.6,
        best_eval_loss=None,
        train_config={"lr": 1e-3},
        run_config={"epochs": 2},
    )

    paths = checkpoint_paths(tmp_path)
    assert best_eval_loss == 0.6
    assert paths["last"].exists()
    assert paths["best"].exists()

    metadata = load_checkpoint_metadata(paths["metadata"])
    assert metadata["last"]["epoch"] == 1
    assert metadata["last"]["global_step"] == 5
    assert metadata["best"]["epoch"] == 1
    assert metadata["best"]["eval_loss"] == 0.6

    payload = torch.load(paths["last"], map_location="cpu")
    assert payload["epoch"] == 1
    assert payload["global_step"] == 5
    assert payload["train_config"] == {"lr": 1e-3}
    assert payload["run_config"] == {"epochs": 2}


def test_save_checkpoint_only_updates_best_when_eval_improves(tmp_path):
    accelerator = FakeAccelerator()
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)

    save_checkpoint(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        run_dir=tmp_path,
        epoch=1,
        global_step=5,
        train_loss=0.8,
        eval_loss=0.6,
        best_eval_loss=None,
        train_config={"lr": 1e-3},
        run_config={"epochs": 3},
    )

    best_eval_loss = save_checkpoint(
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        run_dir=tmp_path,
        epoch=2,
        global_step=10,
        train_loss=0.7,
        eval_loss=0.9,
        best_eval_loss=0.6,
        train_config={"lr": 1e-3},
        run_config={"epochs": 3},
    )

    paths = checkpoint_paths(tmp_path)
    metadata = json.loads(paths["metadata"].read_text())

    assert best_eval_loss == 0.6
    assert metadata["last"]["epoch"] == 2
    assert metadata["last"]["eval_loss"] == 0.9
    assert metadata["best"]["epoch"] == 1
    assert metadata["best"]["eval_loss"] == 0.6

    best_payload = torch.load(paths["best"], map_location="cpu")
    assert best_payload["epoch"] == 1
