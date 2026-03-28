import anndata as ad
import numpy as np
import pytest
from scipy import sparse

from cellkit.data.preprocessing import (
    UNS_N_BINS,
    UNS_PREPROCESSING,
    annotate_preprocessing,
    bin_expression,
    drop_cells_with_no_expression,
    filter_genes,
    log1p_transform,
    normalize_total,
    read_anndata,
    write_anndata,
)


def test_filter_genes_preserves_overlap_order():
    adata = ad.AnnData(
        X=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        var={"gene_name": ["GENE_A", "GENE_B", "GENE_C"]},
    )
    adata.var_names = ["GENE_A", "GENE_B", "GENE_C"]

    filtered = filter_genes(adata, {"GENE_A", "GENE_C"})

    assert list(filtered.var_names) == ["GENE_A", "GENE_C"]
    np.testing.assert_array_equal(filtered.X, np.array([[1.0, 3.0]], dtype=np.float32))


def test_filter_genes_raises_when_no_overlap():
    adata = ad.AnnData(X=np.array([[1.0]], dtype=np.float32))
    adata.var_names = ["GENE_B"]

    with pytest.raises(ValueError, match="No dataset genes overlap"):
        filter_genes(adata, {"GENE_A"})


def test_drop_cells_with_no_expression_removes_empty_rows():
    adata = ad.AnnData(X=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]], dtype=np.float32))

    filtered = drop_cells_with_no_expression(adata)

    assert filtered.n_obs == 2
    np.testing.assert_array_equal(
        filtered.X,
        np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32),
    )


def test_normalize_total_scales_rows_to_target_sum():
    adata = ad.AnnData(X=np.array([[1.0, 3.0], [2.0, 2.0]], dtype=np.float32))

    normalized = normalize_total(adata, target_sum=100.0)

    row_sums = np.asarray(normalized.X.sum(axis=1)).reshape(-1)
    np.testing.assert_allclose(row_sums, np.array([100.0, 100.0], dtype=np.float32))


def test_log1p_transform_supports_sparse_matrices():
    x = sparse.csr_matrix(np.array([[0.0, 1.0], [3.0, 0.0]], dtype=np.float32))
    adata = ad.AnnData(X=x)

    transformed = log1p_transform(adata)

    np.testing.assert_allclose(
        transformed.X.toarray(),
        np.log1p(np.array([[0.0, 1.0], [3.0, 0.0]], dtype=np.float32)),
    )


def test_bin_expression_bins_each_row_independently():
    adata = ad.AnnData(
        X=np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 5.0, 5.0, 5.0],
            ],
            dtype=np.float32,
        )
    )

    binned = bin_expression(adata, n_bins=3)

    np.testing.assert_array_equal(
        binned.X,
        np.array(
            [
                [0, 1, 2, 3],
                [0, 3, 3, 3],
            ],
            dtype=np.uint16,
        ),
    )
    assert binned.uns[UNS_N_BINS] == 3


def test_annotate_preprocessing_records_contract():
    adata = ad.AnnData(X=np.array([[1.0]], dtype=np.float32))

    annotated = annotate_preprocessing(
        adata,
        normalized=True,
        log1p_applied=True,
        n_bins=51,
        dropped_zero_expression_cells=True,
    )

    assert annotated.uns[UNS_PREPROCESSING] == {
        "normalized": True,
        "log1p": True,
        "n_bins": 51,
        "dropped_zero_expression_cells": True,
    }


def test_read_and_write_anndata_round_trip_h5ad(tmp_path):
    path = tmp_path / "dataset.h5ad"
    adata = ad.AnnData(X=np.array([[1.0, 2.0]], dtype=np.float32))
    adata.var_names = ["GENE_A", "GENE_B"]

    write_anndata(adata, path)
    loaded = read_anndata(path)

    np.testing.assert_array_equal(loaded.X, adata.X)
    assert list(loaded.var_names) == ["GENE_A", "GENE_B"]
