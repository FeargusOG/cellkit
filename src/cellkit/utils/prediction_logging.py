from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Any

import torch

PREDICTIONS_DIRNAME = "predictions"
PREDICTIONS_LOG_NAME = "masked_predictions.jsonl"
_STOP = object()


def prediction_log_path(run_dir: Path) -> Path:
    return run_dir / PREDICTIONS_DIRNAME / PREDICTIONS_LOG_NAME


class AsyncPredictionLogger:
    def __init__(
        self,
        output_path: Path,
        *,
        max_samples_per_batch: int,
        queue_size: int = 128,
        buffer_size: int = 32,
    ):
        self.output_path = output_path
        self.max_samples_per_batch = max_samples_per_batch
        self.queue: queue.Queue[dict[str, Any] | object] = queue.Queue(maxsize=queue_size)
        self.buffer_size = buffer_size
        self.dropped_messages = 0
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    def log_batch(
        self,
        *,
        epoch: int,
        batch: int,
        genes: torch.Tensor,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        if self.max_samples_per_batch == 0:
            return
        payload = build_prediction_log_payload(
            epoch=epoch,
            batch=batch,
            genes=genes,
            pred=pred,
            target=target,
            mask=mask,
            max_samples=self.max_samples_per_batch,
        )
        if payload is None:
            return
        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            self.dropped_messages += 1

    def close(self) -> None:
        self.queue.put(_STOP)
        self._thread.join()

    def _writer_loop(self) -> None:
        buffered_lines: list[str] = []
        with self.output_path.open("a", encoding="utf-8") as handle:
            while True:
                item = self.queue.get()
                if item is _STOP:
                    break
                buffered_lines.append(json.dumps(item, sort_keys=True))
                if len(buffered_lines) >= self.buffer_size:
                    handle.write("\n".join(buffered_lines) + "\n")
                    handle.flush()
                    buffered_lines.clear()
            if buffered_lines:
                handle.write("\n".join(buffered_lines) + "\n")
                handle.flush()


def build_prediction_log_payload(
    *,
    epoch: int,
    batch: int,
    genes: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    max_samples: int,
) -> dict[str, Any] | None:
    masked_genes = genes[mask]
    if masked_genes.numel() == 0 or max_samples == 0:
        return None

    masked_pred = pred[mask]
    masked_target = target[mask]
    sample_count = min(max_samples, masked_genes.numel())
    sample_positions = torch.randperm(masked_genes.numel(), device=masked_genes.device)[
        :sample_count
    ]

    sampled_genes = masked_genes[sample_positions].detach().cpu().tolist()
    sampled_pred = masked_pred[sample_positions].detach().cpu().tolist()
    sampled_target = masked_target[sample_positions].detach().cpu().tolist()

    return {
        "epoch": epoch,
        "batch": batch,
        "sample_count": sample_count,
        "samples": [
            {
                "gene_id": int(gene_id),
                "pred": round(float(pred_value), 4),
                "true": round(float(target_value), 4),
            }
            for gene_id, pred_value, target_value in zip(
                sampled_genes, sampled_pred, sampled_target, strict=True
            )
        ],
    }
