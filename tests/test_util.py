import numpy as np
import anndata as ad

from cellkit.data.util import stratified_subsample_adata


def test_stratified_subsample_adata_basic():
    X = np.arange(12, dtype=float).reshape(4, 3)
    adata = ad.AnnData(X=X)
    adata.obs["group"] = ["a", "a", "b", "b"]

    sub = stratified_subsample_adata(
        adata, frac=0.5, strata_cols=["group"], random_state=0
    )

    assert sub.n_obs == 2
    assert set(sub.obs["group"]) <= {"a", "b"}


def test_stratified_subsample_adata_singleton_fallback(capsys):
    X = np.arange(9, dtype=float).reshape(3, 3)
    adata = ad.AnnData(X=X)
    adata.obs["group"] = ["a", "a", "b"]

    sub = stratified_subsample_adata(
        adata,
        frac=0.5,
        strata_cols=["group"],
        random_state=0,
        drop_small_strata=False,
    )

    captured = capsys.readouterr()
    assert "Falling back to unstratified sampling" in captured.out
    assert sub.n_obs == 2


def test_stratified_subsample_adata_singleton_dropped(capsys):
    X = np.arange(9, dtype=float).reshape(3, 3)
    adata = ad.AnnData(X=X)
    adata.obs["group"] = ["a", "a", "b"]

    sub = stratified_subsample_adata(
        adata,
        frac=0.5,
        strata_cols=["group"],
        random_state=0,
        drop_small_strata=True,
    )

    captured = capsys.readouterr()
    assert "Falling back to unstratified sampling" not in captured.out
    assert sub.n_obs == 1
    assert set(sub.obs["group"]) == {"a"}


def test_stratified_subsample_adata_unstratified():
    X = np.arange(20, dtype=float).reshape(5, 4)
    adata = ad.AnnData(X=X)
    adata.obs["group"] = ["a", "b", "c", "d", "e"]

    sub = stratified_subsample_adata(adata, frac=0.4, strata_cols=None, random_state=0)

    assert sub.n_obs == 2


def test_stratified_subsample_adata_multiple_strata_cols():
    X = np.arange(24, dtype=float).reshape(6, 4)
    adata = ad.AnnData(X=X)
    adata.obs["donor"] = ["d1", "d1", "d1", "d2", "d2", "d2"]
    adata.obs["batch"] = ["b1", "b1", "b2", "b1", "b2", "b2"]

    sub = stratified_subsample_adata(
        adata,
        frac=0.5,
        strata_cols=["donor", "batch"],
        random_state=0,
        drop_small_strata=False,
    )

    assert sub.n_obs == 3


def test_stratified_subsample_adata_drop_small_strata_noop(capsys):
    X = np.arange(12, dtype=float).reshape(4, 3)
    adata = ad.AnnData(X=X)
    adata.obs["group"] = ["a", "a", "b", "b"]

    sub = stratified_subsample_adata(
        adata,
        frac=0.5,
        strata_cols=["group"],
        random_state=0,
        drop_small_strata=True,
    )

    captured = capsys.readouterr()
    assert "Falling back to unstratified sampling" not in captured.out
    assert sub.n_obs == 2


def test_stratified_subsample_adata_drop_small_strata_default():
    X = np.arange(9, dtype=float).reshape(3, 3)
    adata = ad.AnnData(X=X)
    adata.obs["group"] = ["a", "a", "b"]

    sub = stratified_subsample_adata(
        adata, frac=0.5, strata_cols=["group"], random_state=0
    )

    assert sub.n_obs == 1
    assert set(sub.obs["group"]) == {"a"}


def test_stratified_subsample_adata_preserves_obs_index_labels():
    X = np.arange(30, dtype=float).reshape(6, 5)
    adata = ad.AnnData(X=X)
    adata.obs.index = ["c3", "c1", "c2", "c6", "c5", "c4"]
    adata.obs["group"] = ["a", "a", "b", "b", "a", "b"]

    sub = stratified_subsample_adata(
        adata, frac=0.5, strata_cols=["group"], random_state=0
    )

    assert set(sub.obs.index).issubset(set(adata.obs.index))
    for label in sub.obs.index:
        assert label in adata.obs.index
