import json

import torch

from cellkit.utils.prediction_logging import (
    AsyncPredictionLogger,
    build_prediction_log_payload,
    prediction_log_path,
)


def test_build_prediction_log_payload_samples_masked_positions():
    torch.manual_seed(0)
    payload = build_prediction_log_payload(
        epoch=2,
        batch=3,
        genes=torch.tensor([[3, 4, 5, 6]]),
        pred=torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
        target=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        mask=torch.tensor([[False, True, True, False]]),
        max_samples=1,
    )

    assert payload is not None
    assert payload["epoch"] == 2
    assert payload["batch"] == 3
    assert payload["sample_count"] == 1
    assert len(payload["samples"]) == 1
    assert payload["samples"][0]["gene_id"] in {4, 5}


def test_async_prediction_logger_writes_chunked_jsonl(tmp_path):
    torch.manual_seed(0)
    output_path = prediction_log_path(tmp_path)
    logger = AsyncPredictionLogger(output_path, max_samples_per_batch=2, buffer_size=2)

    logger.log_batch(
        epoch=1,
        batch=0,
        genes=torch.tensor([[3, 4, 5]]),
        pred=torch.tensor([[0.1, 0.2, 0.3]]),
        target=torch.tensor([[1.0, 2.0, 3.0]]),
        mask=torch.tensor([[True, False, True]]),
    )
    logger.log_batch(
        epoch=1,
        batch=1,
        genes=torch.tensor([[6, 7, 8]]),
        pred=torch.tensor([[0.4, 0.5, 0.6]]),
        target=torch.tensor([[4.0, 5.0, 6.0]]),
        mask=torch.tensor([[False, True, True]]),
    )
    logger.close()

    lines = output_path.read_text().strip().splitlines()
    assert len(lines) == 2
    payload = json.loads(lines[0])
    assert payload["epoch"] == 1
    assert "samples" in payload


def test_async_prediction_logger_drops_messages_when_queue_is_full(tmp_path):
    output_path = prediction_log_path(tmp_path)
    logger = AsyncPredictionLogger(
        output_path,
        max_samples_per_batch=1,
        queue_size=1,
        buffer_size=10,
    )
    logger.queue.put_nowait(
        {
            "epoch": 0,
            "batch": 0,
            "sample_count": 1,
            "samples": [{"gene_id": 3, "pred": 0.1, "true": 1.0}],
        }
    )

    logger.log_batch(
        epoch=1,
        batch=1,
        genes=torch.tensor([[3, 4]]),
        pred=torch.tensor([[0.1, 0.2]]),
        target=torch.tensor([[1.0, 2.0]]),
        mask=torch.tensor([[True, False]]),
    )
    logger.close()

    assert logger.dropped_messages == 1
