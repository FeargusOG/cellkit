import numpy as np
from scipy.sparse import issparse


def _count_nonzero(X, axis):
    if issparse(X):
        return (X > 0).sum(axis=axis).A1
    return (X > 0).sum(axis=axis)


def _resolve_layer(adata, layer):
    if layer is None or layer == "X":
        return adata.X, "X", False
    if layer not in adata.layers:
        return None, layer, True
    return adata.layers[layer], layer, False


def _summary_stats(values):
    values = np.asarray(values).ravel()
    if values.size == 0:
        nan = float("nan")
        return {"min": nan, "max": nan, "mean": nan, "median": nan}
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
    }


def compute_expression_coverage(adata, layer=None):
    X, label, missing = _resolve_layer(adata, layer)
    if missing:
        return {"label": label, "missing": True}

    genes_per_cell = _count_nonzero(X, axis=1)
    cells_per_gene = _count_nonzero(X, axis=0)
    return {
        "label": label,
        "missing": False,
        "genes_per_cell": _summary_stats(genes_per_cell),
        "cells_per_gene": _summary_stats(cells_per_gene),
    }


def compute_value_stats(adata, layer=None):
    X, label, missing = _resolve_layer(adata, layer)
    if missing:
        return {"label": label, "missing": True}
    assert X is not None

    if issparse(X):
        X = X.tocoo()
        if X.nnz == 0:
            return {
                "label": label,
                "missing": False,
                "sparse": True,
                "nnz": 0,
                "stats": None,
            }
        return {
            "label": label,
            "missing": False,
            "sparse": True,
            "nnz": int(X.nnz),
            "stats": _summary_stats(X.data),
        }

    return {
        "label": label,
        "missing": False,
        "sparse": False,
        "nnz": None,
        "stats": _summary_stats(X),
    }


def summarize_expression_coverage(adata, layer=None):
    """
    Print basic expression coverage statistics for an AnnData object.

    Reports:
    - Number of genes with non-zero expression per cell
    - Number of cells with non-zero expression per gene
    """
    result = compute_expression_coverage(adata, layer=layer)
    if result["missing"]:
        print(f"\nWarning: layer '{result['label']}' not found. Skipping.")
        return

    print(f"\nExpression coverage for layer: {result['label']}")
    genes_per_cell = result["genes_per_cell"]
    print("\nPer-cell expression coverage:")
    print(f"Mean genes per cell: {genes_per_cell['mean']:.2f}")
    print(f"Median genes per cell: {genes_per_cell['median']:.2f}")
    print(
        f"Min/Max genes per cell: " f"{genes_per_cell['min']} / {genes_per_cell['max']}"
    )

    # ---- Per-gene: cells with non-zero expression ----
    cells_per_gene = result["cells_per_gene"]
    print("\nPer-gene expression coverage:")
    print(f"Mean cells per gene: {cells_per_gene['mean']:.2f}")
    print(f"Median cells per gene: {cells_per_gene['median']:.2f}")
    print(
        f"Min/Max cells per gene: " f"{cells_per_gene['min']} / {cells_per_gene['max']}"
    )


def summarize_value_stats(adata, layer=None):
    result = compute_value_stats(adata, layer=layer)
    if result["missing"]:
        print(f"\nWarning: layer '{result['label']}' not found. Skipping.")
        return

    print(f"\nValue summary for layer: {result['label']}")
    if result["sparse"]:
        print(f"  Non-zero entries: {result['nnz']}")
        if result["stats"] is None:
            print("  Min (non-zero): n/a (no non-zero entries)")
            print("  Max (non-zero): n/a (no non-zero entries)")
            print("  Mean (non-zero): n/a (no non-zero entries)")
            print("  Median (non-zero): n/a (no non-zero entries)")
        else:
            stats = result["stats"]
            print(f"  Min (non-zero): {stats['min']}")
            print(f"  Max (non-zero): {stats['max']}")
            print(f"  Mean (non-zero): {stats['mean']:.3f}")
            print(f"  Median (non-zero): {stats['median']:.3f}")
    else:
        stats = result["stats"]
        print(f"  Min: {stats['min']}")
        print(f"  Max: {stats['max']}")
        print(f"  Mean: {stats['mean']:.3f}")
        print(f"  Median: {stats['median']:.3f}")


def get_anndata_summary(adata):
    """
    Return a lightweight, machine-readable summary of an AnnData object.
    Can be used for logging our dataset at the start of a run.
    """

    X = adata.X

    summary = {
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
        "X_type": type(X).__name__,
        "X_is_sparse": issparse(X),
        "layers": sorted(adata.layers.keys()),
        "obs_columns": sorted(adata.obs.columns),
        "var_columns": sorted(adata.var.columns),
        "uns_keys": sorted(adata.uns.keys()),
    }

    # Optional: shape sanity
    summary["X_shape"] = X.shape

    return summary
