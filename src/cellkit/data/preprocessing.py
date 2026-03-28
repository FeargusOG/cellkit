from __future__ import annotations

from collections.abc import Collection
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

UNS_N_BINS = "n_bins"
UNS_PREPROCESSING = "preprocessing"


def read_anndata(path: str | Path) -> ad.AnnData:
    path = Path(path)
    if path.suffix == ".h5ad":
        return ad.read_h5ad(path)
    if path.suffix == ".zarr":
        return ad.read_zarr(path)
    raise ValueError("Unsupported AnnData format; expected .h5ad or .zarr")


def write_anndata(adata: ad.AnnData, path: str | Path) -> None:
    path = Path(path)
    if path.suffix == ".h5ad":
        adata.write_h5ad(path)
        return
    if path.suffix == ".zarr":
        adata.write_zarr(path)
        return
    raise ValueError("Unsupported AnnData format; expected .h5ad or .zarr")


def filter_genes(adata: ad.AnnData, genes_to_keep: Collection[str]) -> ad.AnnData:
    retained_genes = [gene for gene in adata.var_names if gene in genes_to_keep]
    if not retained_genes:
        raise ValueError("No dataset genes overlap with the provided gene set")
    return adata[:, retained_genes].copy()


def drop_cells_with_no_expression(adata: ad.AnnData) -> ad.AnnData:
    row_sums = _row_sums(adata.X)
    keep_mask = row_sums > 0
    if not np.any(keep_mask):
        raise ValueError("No cells retain expression after feature filtering")
    return adata[keep_mask].copy()


def normalize_total(adata: ad.AnnData, *, target_sum: float) -> ad.AnnData:
    row_sums = _row_sums(adata.X)
    scale = np.divide(
        target_sum,
        row_sums,
        out=np.zeros_like(row_sums, dtype=np.float32),
        where=row_sums > 0,
    )
    if sparse.issparse(adata.X):
        adata.X = sparse.diags(scale) @ adata.X
    else:
        adata.X = adata.X * scale[:, None]
    adata.X = adata.X.astype(np.float32)
    return adata


def log1p_transform(adata: ad.AnnData) -> ad.AnnData:
    if sparse.issparse(adata.X):
        adata.X.data = np.log1p(adata.X.data)
    else:
        adata.X = np.log1p(adata.X)
    adata.X = adata.X.astype(np.float32)
    return adata


def bin_expression(adata: ad.AnnData, *, n_bins: int) -> ad.AnnData:
    if n_bins <= 1:
        raise ValueError("n_bins must be > 1")

    if sparse.issparse(adata.X):
        x = adata.X.tocsr()
        binned = sparse.lil_matrix(x.shape, dtype=np.uint16)
        for row in range(x.shape[0]):
            values = x.getrow(row).toarray().reshape(-1)
            binned[row] = _bin_row(values, n_bins=n_bins)
        adata.X = binned.tocsr()
    else:
        x = np.asarray(adata.X, dtype=np.float32)
        binned = np.zeros_like(x, dtype=np.uint16)
        for row in range(x.shape[0]):
            binned[row] = _bin_row(x[row], n_bins=n_bins)
        adata.X = binned

    adata.uns[UNS_N_BINS] = n_bins
    return adata


def annotate_preprocessing(
    adata: ad.AnnData,
    *,
    normalized: bool,
    log1p_applied: bool,
    n_bins: int | None,
    dropped_zero_expression_cells: bool,
) -> ad.AnnData:
    adata.uns[UNS_PREPROCESSING] = {
        "normalized": normalized,
        "log1p": log1p_applied,
        "n_bins": n_bins,
        "dropped_zero_expression_cells": dropped_zero_expression_cells,
    }
    return adata


def _row_sums(x) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.sum(axis=1)).reshape(-1).astype(np.float32)
    return np.asarray(x.sum(axis=1), dtype=np.float32).reshape(-1)


def _bin_row(values: np.ndarray, *, n_bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    binned = np.zeros(values.shape, dtype=np.uint16)
    nonzero_mask = values > 0
    if not np.any(nonzero_mask):
        return binned

    nonzero_values = values[nonzero_mask]
    if np.all(nonzero_values == nonzero_values[0]):
        binned[nonzero_mask] = n_bins
        return binned

    quantiles = np.quantile(nonzero_values, q=np.linspace(0, 1, n_bins + 1)[1:-1])
    bin_ids = np.digitize(nonzero_values, quantiles, right=False) + 1
    binned[nonzero_mask] = bin_ids.astype(np.uint16)
    return binned
