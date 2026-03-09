import numpy as np
import anndata as ad
from scipy.sparse import csr_matrix

from cellkit.data.explore import compute_expression_coverage, compute_value_stats
from cellkit.data.explore import get_anndata_summary


def test_compute_expression_coverage_dense():
    X = np.array([[0, 1, 2], [0, 0, 3]], dtype=float)
    adata = ad.AnnData(X=X)

    result = compute_expression_coverage(adata)

    assert result["missing"] is False
    genes_per_cell = result["genes_per_cell"]
    cells_per_gene = result["cells_per_gene"]

    assert genes_per_cell["mean"] == 1.5
    assert genes_per_cell["median"] == 1.5
    assert genes_per_cell["min"] == 1.0
    assert genes_per_cell["max"] == 2.0

    assert cells_per_gene["mean"] == 1.0
    assert cells_per_gene["median"] == 1.0
    assert cells_per_gene["min"] == 0.0
    assert cells_per_gene["max"] == 2.0


def test_compute_value_stats_dense():
    X = np.array([[0, 1, 2], [0, 0, 3]], dtype=float)
    adata = ad.AnnData(X=X)

    result = compute_value_stats(adata)

    assert result["missing"] is False
    assert result["sparse"] is False
    stats = result["stats"]
    assert stats["min"] == 0.0
    assert stats["max"] == 3.0
    assert stats["mean"] == 1.0
    assert stats["median"] == 0.5


def test_compute_value_stats_sparse_empty():
    X = csr_matrix((2, 3))
    adata = ad.AnnData(X=X)

    result = compute_value_stats(adata)

    assert result["missing"] is False
    assert result["sparse"] is True
    assert result["nnz"] == 0
    assert result["stats"] is None


def test_missing_layer():
    X = np.array([[1, 0], [0, 1]], dtype=float)
    adata = ad.AnnData(X=X)

    result = compute_expression_coverage(adata, layer="missing")
    assert result["missing"] is True
    assert result["label"] == "missing"


def test_get_anndata_summary_sorted_lists():
    X = np.array([[1, 0], [0, 1]], dtype=float)
    adata = ad.AnnData(X=X)
    adata.obs["b"] = [1, 2]
    adata.obs["a"] = [3, 4]
    adata.var["z"] = [0.1, 0.2]
    adata.var["y"] = [0.3, 0.4]
    adata.uns["key_b"] = 1
    adata.uns["key_a"] = 2
    adata.layers["layer_b"] = X
    adata.layers["layer_a"] = X

    summary = get_anndata_summary(adata)

    assert summary["n_cells"] == 2
    assert summary["n_genes"] == 2
    assert summary["X_is_sparse"] is False
    assert summary["layers"] == ["layer_a", "layer_b"]
    assert summary["obs_columns"] == ["a", "b"]
    assert summary["var_columns"] == ["y", "z"]
    assert summary["uns_keys"] == ["key_a", "key_b"]
