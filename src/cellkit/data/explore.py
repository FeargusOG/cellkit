import numpy as np
from scipy.sparse import issparse


def _count_nonzero(X, axis):
    if issparse(X):
        return (X > 0).sum(axis=axis).A1
    return (X > 0).sum(axis=axis)

def _resolve_layer(adata, layer):
    if layer is None or layer == "X":
        return adata.X, "X"
    if layer not in adata.layers:
        return None, layer
    return adata.layers[layer], layer


def summarize_expression_coverage(adata, layer=None):
    """
    Print basic expression coverage statistics for an AnnData object.

    Reports:
    - Number of genes with non-zero expression per cell
    - Number of cells with non-zero expression per gene
    """
    X, label = _resolve_layer(adata, layer)
    if X is None:
        print(f"\nWarning: layer '{label}' not found. Skipping.")
        return

    print(f"\nExpression coverage for layer: {label}")

    # ---- Per-cell: genes with non-zero expression ----
    genes_per_cell = _count_nonzero(X, axis=1)
    print("\nPer-cell expression coverage:")
    print(f"Mean genes per cell: {genes_per_cell.mean():.2f}")
    print(f"Median genes per cell: {np.median(genes_per_cell):.2f}")
    print(
        f"Min/Max genes per cell: " f"{genes_per_cell.min()} / {genes_per_cell.max()}"
    )

    # ---- Per-gene: cells with non-zero expression ----
    cells_per_gene = _count_nonzero(X, axis=0)

    print("\nPer-gene expression coverage:")
    print(f"Mean cells per gene: {cells_per_gene.mean():.2f}")
    print(f"Median cells per gene: {np.median(cells_per_gene):.2f}")
    print(
        f"Min/Max cells per gene: " f"{cells_per_gene.min()} / {cells_per_gene.max()}"
    )


def summarize_value_stats(adata, layer=None):
    X, label = _resolve_layer(adata, layer)
    if X is None:
        print(f"\nWarning: layer '{label}' not found. Skipping.")
        return

    print(f"\nValue summary for layer: {label}")
    if issparse(X):
        X = X.tocoo()
        print(f"  Non-zero entries: {X.nnz}")
        if X.nnz == 0:
            print("  Min (non-zero): n/a (no non-zero entries)")
            print("  Max (non-zero): n/a (no non-zero entries)")
            print("  Mean (non-zero): n/a (no non-zero entries)")
            print("  Median (non-zero): n/a (no non-zero entries)")
        else:
            print(f"  Min (non-zero): {X.data.min()}")
            print(f"  Max (non-zero): {X.data.max()}")
            print(f"  Mean (non-zero): {X.data.mean():.3f}")
            print(f"  Median (non-zero): {np.median(X.data):.3f}")
    else:
        print(f"  Min: {X.min()}")
        print(f"  Max: {X.max()}")
        print(f"  Mean: {X.mean():.3f}")
        print(f"  Median: {np.median(X):.3f}")
