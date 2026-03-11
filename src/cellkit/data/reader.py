"""Lazy reader abstractions for disk-backed cell datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
from anndata import experimental as ad_experimental


class DataReader(ABC):
    """Interface for lazily reading indexed samples from disk-backed datasets."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples available through the reader."""

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        """Return the dataset shape as ``(num_rows, num_features)``."""

    @property
    @abstractmethod
    def var_names(self) -> list[str]:
        """Return feature names in dataset order."""

    @property
    @abstractmethod
    def obs_columns(self) -> list[str]:
        """Return available observation metadata columns."""

    @abstractmethod
    def read_x(self, index: int, layer: str | None = None) -> np.ndarray:
        """Read one feature row from ``X`` or from a named layer."""

    @abstractmethod
    def read_obs(self, index: int, columns: list[str] | None = None) -> dict[str, Any]:
        """Read one observation row as a plain dictionary."""

    @abstractmethod
    def close(self) -> None:
        """Close any open file handles held by the reader."""


class AnnDataReader(DataReader, ABC):
    """Base class for lazy AnnData-backed readers."""

    def __init__(self, path: str | PathLike[str]):
        """Initialize the reader with a path to an AnnData-backed store.

        Args:
            path: Filesystem path to a ``.h5ad`` or ``.zarr`` dataset.
        """
        self.path = Path(path)
        self._adata: ad.AnnData | None = None

    def __len__(self) -> int:
        """Return the number of observations in the dataset."""
        return int(self._ensure_open().n_obs)

    @property
    def shape(self) -> tuple[int, int]:
        """Return the AnnData matrix shape as ``(n_obs, n_vars)``."""
        adata = self._ensure_open()
        return int(adata.n_obs), int(adata.n_vars)

    @property
    def var_names(self) -> list[str]:
        """Return feature names in dataset order."""
        return [str(name) for name in self._ensure_open().var_names]

    @property
    def obs_columns(self) -> list[str]:
        """Return available observation metadata columns."""
        return [str(column) for column in self._ensure_open().obs.columns]

    def read_x(self, index: int, layer: str | None = None) -> np.ndarray:
        """Read one feature row from ``X`` or from a named layer.

        Args:
            index: Zero-based observation index.
            layer: Optional layer name. If omitted, reads from ``adata.X``.

        Returns:
            Dense one-dimensional NumPy array containing the requested row. The
            returned dtype is preserved from the underlying stored data rather than
            coerced to a fixed dtype such as ``float32``.
        """
        adata = self._ensure_open()
        matrix = adata.X if layer is None else adata.layers[layer]
        if matrix is None:
            raise ValueError("Requested matrix is missing")
        row = matrix[index]
        to_memory = getattr(row, "to_memory", None)
        if callable(to_memory):
            row = to_memory()
        to_array = getattr(row, "toarray", None)
        if callable(to_array):
            row = to_array()
        row = np.asarray(row)
        if row.ndim == 2:
            if row.shape[0] != 1:
                raise ValueError(f"Expected a single row, got shape {row.shape}")
            row = row[0]
        elif row.ndim != 1:
            raise ValueError(f"Expected a 1D row, got shape {row.shape}")
        return row

    def read_obs(self, index: int, columns: list[str] | None = None) -> dict[str, Any]:
        """Read one observation row as a plain dictionary.

        Args:
            index: Zero-based observation index.
            columns: Optional subset of ``obs`` columns to return.

        Returns:
            Dictionary mapping column names to scalar values.
        """
        obs = self._ensure_open().obs
        if columns is None:
            selected_obs = obs
        else:
            selected_obs = obs[columns]

        obs_row = selected_obs.iloc[index]

        to_pandas = getattr(obs_row, "to_pandas", None)
        if callable(to_pandas):
            obs_row = to_pandas()

        values = np.asarray(obs_row).reshape(-1)
        column_names = [str(column) for column in selected_obs.columns]
        return dict(zip(column_names, values, strict=True))

    def close(self) -> None:
        """Close any open backing store if the current AnnData object exposes one."""
        if self._adata is None:
            return
        file_manager = getattr(self._adata, "file", None)
        if file_manager is not None and hasattr(file_manager, "close"):
            file_manager.close()
        self._adata = None

    def __getstate__(self) -> dict[str, Any]:
        """Drop open AnnData handles before pickling the reader."""
        state = self.__dict__.copy()
        state["_adata"] = None
        return state

    def _ensure_open(self) -> ad.AnnData:
        """Open the AnnData object on first access and cache it locally."""
        if self._adata is None:
            self._adata = self._open()
        return self._adata

    @abstractmethod
    def _open(self) -> ad.AnnData:
        """Open the backing AnnData store lazily."""


class H5ADReader(AnnDataReader):
    """Reader for ``.h5ad`` files using backed read mode."""

    def _open(self) -> ad.AnnData:
        """Open the ``.h5ad`` file without materializing the full matrix."""
        return ad.read_h5ad(self.path, backed="r")


class ZarrReader(AnnDataReader):
    """Reader for ``.zarr`` AnnData stores using lazy loading."""

    def _open(self) -> ad.AnnData:
        """Open the ``.zarr`` store lazily."""
        return ad_experimental.read_lazy(self.path) # pyright: ignore[reportCallIssue]
