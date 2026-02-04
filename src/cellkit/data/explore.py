import numpy as np
from scipy.sparse import issparse


def _count_nonzero(X, axis):
    if issparse(X):
        return (X > 0).sum(axis=axis).A1
    return (X > 0).sum(axis=axis)


def summarize_expression_coverage(adata):
    """
    Print basic expression coverage statistics for an AnnData object.

    Reports:
    - Number of genes with non-zero expression per cell
    - Number of cells with non-zero expression per gene
    """

    # ---- Per-cell: genes with non-zero expression ----
    genes_per_cell = _count_nonzero(adata.X, axis=1)
    print("\nPer-cell expression coverage:")
    print(f"Mean genes per cell: {genes_per_cell.mean():.2f}")
    print(f"Median genes per cell: {np.median(genes_per_cell):.2f}")
    print(
        f"Min/Max genes per cell: " f"{genes_per_cell.min()} / {genes_per_cell.max()}"
    )

    # ---- Per-gene: cells with non-zero expression ----
    cells_per_gene = _count_nonzero(adata.X, axis=0)
    if issparse(adata.X):
        cells_per_gene = (adata.X > 0).sum(axis=0).A1
    else:
        cells_per_gene = (adata.X > 0).sum(axis=0)

    print("\nPer-gene expression coverage:")
    print(f"Mean cells per gene: {cells_per_gene.mean():.2f}")
    print(f"Median cells per gene: {np.median(cells_per_gene):.2f}")
    print(
        f"Min/Max cells per gene: " f"{cells_per_gene.min()} / {cells_per_gene.max()}"
    )
