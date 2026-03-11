"""Map-style PyTorch datasets backed by lazy cell data readers."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import torch

from cellkit.data.reader import DataReader


class AnnDataDataset(torch.utils.data.Dataset):
    """Map-style dataset that loads one AnnData observation at a time."""

    def __init__(
        self,
        reader_factory: Callable[[], DataReader],
        *,
        indices: Sequence[int] | None = None,
        layer: str | None = None,
        obs_columns: Sequence[str] | None = None,
        target_column: str | None = None,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        return_index: bool = False,
    ):
        """Initialize the dataset.

        Args:
            reader_factory: Callable that constructs a new lazy reader instance.
                Each dataset copy creates and caches its own reader on first access.
            indices: Optional subset of observation indices exposed by the dataset.
            layer: Optional layer name used for feature reads instead of ``X``.
            obs_columns: Optional subset of ``obs`` columns to attach to each sample.
            target_column: Optional ``obs`` column to expose separately as ``target``.
            transform: Optional callable applied to each sample dictionary.
            return_index: Whether to include the original observation index in samples.
        """
        self.reader_factory = reader_factory
        self.layer = layer
        self.obs_columns = list(obs_columns) if obs_columns is not None else None
        self.target_column = target_column
        self.transform = transform
        self.return_index = return_index
        self._reader: DataReader | None = None

        preview_reader = self.reader_factory()
        try:
            dataset_length = len(preview_reader)
            if indices is None:
                self.indices = None
                self._length = dataset_length
            else:
                self.indices = list(indices)
                self._length = len(self.indices)
                self._validate_indices(self.indices, dataset_length)

            self._validate_schema(preview_reader)
        finally:
            preview_reader.close()

        if self.obs_columns is not None and target_column is not None:
            if target_column in self.obs_columns:
                raise ValueError(
                    "target_column must not also appear in obs_columns; it is returned separately"
                )

    def __len__(self) -> int:
        """Return the number of samples exposed by the dataset."""
        return self._length

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Read one sample by index.

        Args:
            index: Zero-based dataset index after any subsetting.

        Returns:
            Dictionary containing at least ``x`` and optionally ``obs``, ``target``,
            and ``index``.
        """
        if index < 0 or index >= len(self):
            raise IndexError("dataset index out of range")

        reader = self._get_reader()
        source_index = index if self.indices is None else self.indices[index]

        sample: dict[str, Any] = {
            "x": self._to_tensor(reader.read_x(source_index, layer=self.layer))
        }
        obs_columns = self._requested_obs_columns()
        if obs_columns is not None:
            obs = reader.read_obs(source_index, columns=obs_columns)
            if self.target_column is not None:
                sample["target"] = obs.pop(self.target_column)
            if self.obs_columns is not None:
                sample["obs"] = obs
        if self.return_index:
            sample["index"] = source_index
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def close(self) -> None:
        """Close the worker-local reader if it has been opened."""
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def _get_reader(self) -> DataReader:
        """Return the current worker-local reader, creating it on first use."""
        if self._reader is None:
            self._reader = self.reader_factory()
        return self._reader

    def _requested_obs_columns(self) -> list[str] | None:
        """Return the union of requested obs columns and target column."""
        columns: list[str] = []
        if self.obs_columns is not None:
            columns.extend(self.obs_columns)
        if self.target_column is not None:
            columns.append(self.target_column)
        return columns or None

    @staticmethod
    def _to_tensor(x: np.ndarray) -> torch.Tensor:
        """Convert one dense feature row to a PyTorch tensor."""
        return torch.as_tensor(np.asarray(x))

    @staticmethod
    def _validate_indices(indices: list[int], dataset_length: int) -> None:
        """Validate source indices against the available dataset length."""
        for source_index in indices:
            if source_index < 0 or source_index >= dataset_length:
                raise ValueError(
                    f"indices must be within [0, {dataset_length}), got {source_index}"
                )

    def _validate_schema(self, reader: DataReader) -> None:
        """Validate requested layer and observation metadata names."""
        if self.layer is not None and self.layer not in reader.layer_names:
            raise ValueError(f"Unknown layer '{self.layer}'")

        available_obs_columns = set(reader.obs_columns)
        if self.obs_columns is not None:
            missing_columns = [
                column
                for column in self.obs_columns
                if column not in available_obs_columns
            ]
            if missing_columns:
                raise ValueError(f"Unknown obs columns: {missing_columns}")

        if (
            self.target_column is not None
            and self.target_column not in available_obs_columns
        ):
            raise ValueError(f"Unknown target_column '{self.target_column}'")
