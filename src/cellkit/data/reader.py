"""Lazy reader abstractions for disk-backed cell datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
from typing import Any, cast

import anndata as ad
import numpy as np
from anndata import experimental as ad_experimental
from scipy import sparse


class DataReader(ABC):
    """Interface for lazily reading indexed samples from disk-backed datasets."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples available through the reader."""

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        """Return the dataset shape as ``(num_rows, num_features)``."""

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
        self._adata: Any | None = None

    def __len__(self) -> int:
        """Return the number of observations in the dataset."""
        return int(self._ensure_open().n_obs)

    @property
    def shape(self) -> tuple[int, int]:
        """Return the AnnData matrix shape as ``(n_obs, n_vars)``."""
        adata = self._ensure_open()
        return int(adata.n_obs), int(adata.n_vars)

    def read_x(self, index: int, layer: str | None = None) -> np.ndarray:
        """Read one feature row from ``X`` or from a named layer.

        Args:
            index: Zero-based observation index.
            layer: Optional layer name. If omitted, reads from ``adata.X``.

        Returns:
            Dense one-dimensional NumPy array containing the requested row.
        """
        adata = self._ensure_open()
        matrix = adata.X if layer is None else adata.layers[layer]
        row = matrix[index]
        if hasattr(row, "to_memory"):
            row = row.to_memory()
        if sparse.issparse(row):
            row = row.toarray()
        row = np.asarray(row)
        if row.ndim > 1:
            row = np.squeeze(row, axis=0)
        return row

    def read_obs(self, index: int, columns: list[str] | None = None) -> dict[str, Any]:
        """Read one observation row as a plain dictionary.

        Args:
            index: Zero-based observation index.
            columns: Optional subset of ``obs`` columns to return.

        Returns:
            Dictionary mapping column names to scalar values.
        """
        obs_row = self._ensure_open().obs.iloc[index]
        if columns is not None:
            obs_row = obs_row.loc[columns]
        return obs_row.to_dict()

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

    def _ensure_open(self) -> Any:
        """Open the AnnData object on first access and cache it locally."""
        if self._adata is None:
            self._adata = self._open()
        return self._adata

    @abstractmethod
    def _open(self) -> Any:
        """Open the backing AnnData store lazily."""


class H5ADReader(AnnDataReader):
    """Reader for ``.h5ad`` files using backed read mode."""

    def _open(self) -> Any:
        """Open the ``.h5ad`` file without materializing the full matrix."""
        return ad.read_h5ad(self.path, backed="r")


class ZarrReader(AnnDataReader):
    """Reader for ``.zarr`` AnnData stores using lazy loading."""

    def _open(self) -> Any:
        """Open the ``.zarr`` store lazily."""
        read_lazy = cast(Any, ad_experimental.read_lazy)
        return read_lazy(self.path)


def open_reader(path: str | PathLike[str]) -> DataReader:
    """Create a reader instance based on the dataset file extension.

    Args:
        path: Filesystem path to a supported AnnData-backed store.

    Returns:
        Reader instance configured for the detected storage format.

    Raises:
        ValueError: If the file extension is not supported.
    """
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()
    if suffix == ".h5ad":
        return H5ADReader(path_obj)
    if suffix == ".zarr":
        return ZarrReader(path_obj)
    raise ValueError("Unsupported dataset format; expected .h5ad or .zarr")
