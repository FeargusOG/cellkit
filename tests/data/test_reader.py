from pathlib import Path
import pickle
from typing import Any
from unittest import mock

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from cellkit.data.reader import H5ADReader
from cellkit.data.reader import ZarrReader
from cellkit.data.reader import make_h5ad_reader_factory
from cellkit.data.reader import make_reader_factory
from cellkit.data.reader import make_zarr_reader_factory


class FakeAnnData:
    def __init__(self):
        self.X: Any = np.arange(12, dtype=np.float32).reshape(3, 4)
        self.layers = {
            "counts": sparse.csr_matrix(np.arange(12, dtype=np.float32).reshape(3, 4))
        }
        self.obs = pd.DataFrame(
            {"cell_type": ["a", "b", "c"], "batch": [0, 1, 1]},
            index=["cell0", "cell1", "cell2"],
        )
        self.obs_names = self.obs.index
        self.var_names = pd.Index(["gene_b", "gene_a", "gene_c"])
        self.n_obs = 3
        self.n_vars = 4
        self.file = mock.Mock()


class BadRowMatrix:
    def __getitem__(self, index):
        return np.arange(6, dtype=np.float32).reshape(2, 3)


def make_real_adata() -> ad.AnnData:
    return ad.AnnData(
        X=np.arange(12, dtype=np.float32).reshape(3, 4),
        obs=pd.DataFrame(
            {"cell_type": ["a", "b", "c"], "batch": [0, 1, 1]},
            index=["cell0", "cell1", "cell2"],
        ),
        var=pd.DataFrame(index=["gene_b", "gene_a", "gene_c", "gene_d"]),
    )


def test_h5ad_reader_uses_backed_mode_and_reads_rows():
    fake_adata = FakeAnnData()
    with mock.patch("cellkit.data.reader.ad.read_h5ad", return_value=fake_adata) as read:
        reader = H5ADReader("dataset.h5ad")

        assert len(reader) == 3
        assert reader.shape == (3, 4)
        assert reader.var_names == ["gene_b", "gene_a", "gene_c"]
        assert reader.obs_columns == ["cell_type", "batch"]
        assert reader.obs_names == ["cell0", "cell1", "cell2"]
        assert reader.layer_names == ["counts"]
        np.testing.assert_array_equal(reader.read_x(1), np.array([4, 5, 6, 7]))
        np.testing.assert_array_equal(
            reader.read_x(2, layer="counts"), np.array([8, 9, 10, 11], dtype=np.float32)
        )
        assert reader.read_obs(0, columns=["cell_type"]) == {"cell_type": "a"}

        read.assert_called_once_with(Path("dataset.h5ad"), backed="r")
        reader.close()
        fake_adata.file.close.assert_called_once()


def test_zarr_reader_uses_lazy_open():
    fake_adata = FakeAnnData()
    with mock.patch(
        "cellkit.data.reader.ad_experimental.read_lazy", return_value=fake_adata
    ) as read_lazy:
        reader = ZarrReader("dataset.zarr")

        assert reader.var_names == ["gene_b", "gene_a", "gene_c"]
        assert reader.obs_columns == ["cell_type", "batch"]
        assert reader.obs_names == ["cell0", "cell1", "cell2"]
        assert reader.layer_names == ["counts"]
        np.testing.assert_array_equal(reader.read_x(0), np.array([0, 1, 2, 3]))
        assert reader.read_obs(2, columns=["batch"]) == {"batch": 1}
        assert reader.read_obs(1, columns=["batch", "cell_type"]) == {
            "batch": 1,
            "cell_type": "b",
        }

        read_lazy.assert_called_once_with(Path("dataset.zarr"))


def test_reader_raises_when_x_is_missing():
    fake_adata = FakeAnnData()
    fake_adata.X = None
    with mock.patch("cellkit.data.reader.ad.read_h5ad", return_value=fake_adata):
        reader = H5ADReader("dataset.h5ad")

        with pytest.raises(ValueError, match="Requested matrix is missing"):
            reader.read_x(0)


def test_reader_raises_when_row_shape_is_unexpected():
    fake_adata = FakeAnnData()
    fake_adata.X = BadRowMatrix()
    with mock.patch("cellkit.data.reader.ad.read_h5ad", return_value=fake_adata):
        reader = H5ADReader("dataset.h5ad")

        with pytest.raises(ValueError, match="Expected a single row"):
            reader.read_x(0)


def test_reader_close_is_idempotent():
    fake_adata = FakeAnnData()
    with mock.patch("cellkit.data.reader.ad.read_h5ad", return_value=fake_adata):
        reader = H5ADReader("dataset.h5ad")
        assert len(reader) == 3

        reader.close()
        reader.close()

        fake_adata.file.close.assert_called_once()
        assert reader._adata is None


def test_reader_pickling_drops_open_anndata_handle():
    fake_adata = FakeAnnData()
    with mock.patch("cellkit.data.reader.ad.read_h5ad", return_value=fake_adata):
        reader = H5ADReader("dataset.h5ad")
        assert len(reader) == 3

        restored = pickle.loads(pickle.dumps(reader))

        fake_adata.file.close.assert_not_called()
        assert reader._adata is fake_adata
        assert restored._adata is None


def test_unopened_reader_pickling_preserves_lazy_state():
    reader = H5ADReader("dataset.h5ad")

    restored = pickle.loads(pickle.dumps(reader))

    assert reader._adata is None
    assert restored._adata is None


def test_make_h5ad_reader_factory_creates_new_readers():
    factory = make_h5ad_reader_factory("dataset.h5ad")

    first = factory()
    second = factory()

    assert isinstance(first, H5ADReader)
    assert isinstance(second, H5ADReader)
    assert first is not second
    assert first.path == Path("dataset.h5ad")
    assert second.path == Path("dataset.h5ad")


def test_make_zarr_reader_factory_creates_new_readers():
    factory = make_zarr_reader_factory("dataset.zarr")

    first = factory()
    second = factory()

    assert isinstance(first, ZarrReader)
    assert isinstance(second, ZarrReader)
    assert first is not second
    assert first.path == Path("dataset.zarr")
    assert second.path == Path("dataset.zarr")


def test_make_reader_factory_supports_h5ad():
    factory = make_reader_factory("dataset.h5ad", "h5ad")

    first = factory()
    second = factory()

    assert isinstance(first, H5ADReader)
    assert isinstance(second, H5ADReader)
    assert first is not second
    assert first.path == Path("dataset.h5ad")
    assert second.path == Path("dataset.h5ad")


def test_make_reader_factory_supports_zarr():
    factory = make_reader_factory("dataset.zarr", "zarr")

    first = factory()
    second = factory()

    assert isinstance(first, ZarrReader)
    assert isinstance(second, ZarrReader)
    assert first is not second
    assert first.path == Path("dataset.zarr")
    assert second.path == Path("dataset.zarr")


def test_make_reader_factory_rejects_unknown_format():
    with pytest.raises(ValueError, match="Unsupported data_format"):
        make_reader_factory("dataset.foo", "foo")


def test_h5ad_reader_reads_real_backed_file(tmp_path: Path):
    adata = make_real_adata()
    adata.layers["counts"] = sparse.csr_matrix(
        np.arange(12, dtype=np.float32).reshape(3, 4)
    )
    data_path = tmp_path / "real.h5ad"
    adata.write_h5ad(data_path)

    reader = H5ADReader(data_path)

    assert len(reader) == 3
    assert reader.shape == (3, 4)
    assert reader.var_names == ["gene_b", "gene_a", "gene_c", "gene_d"]
    assert sorted(reader.obs_columns) == ["batch", "cell_type"]
    assert reader.obs_names == ["cell0", "cell1", "cell2"]
    assert reader.layer_names == ["counts"]
    np.testing.assert_array_equal(reader.read_x(1), np.array([4, 5, 6, 7], dtype=np.float32))
    np.testing.assert_array_equal(
        reader.read_x(2, layer="counts"),
        np.array([8, 9, 10, 11], dtype=np.float32),
    )
    assert reader.read_obs(1, columns=["batch", "cell_type"]) == {
        "batch": 1,
        "cell_type": "b",
    }

    reader.close()
    assert reader._adata is None


def test_zarr_reader_reads_real_lazy_store(tmp_path: Path):
    pytest.importorskip("xarray")

    adata = make_real_adata()
    adata.layers["counts"] = np.arange(12, dtype=np.float32).reshape(3, 4)
    data_path = tmp_path / "real.zarr"
    adata.write_zarr(data_path)

    reader = ZarrReader(data_path)

    assert len(reader) == 3
    assert reader.shape == (3, 4)
    assert reader.var_names == ["gene_b", "gene_a", "gene_c", "gene_d"]
    assert sorted(reader.obs_columns) == ["batch", "cell_type"]
    assert reader.obs_names == ["cell0", "cell1", "cell2"]
    assert reader.layer_names == ["counts"]
    np.testing.assert_array_equal(reader.read_x(0), np.array([0, 1, 2, 3], dtype=np.float32))
    np.testing.assert_array_equal(
        reader.read_x(2, layer="counts"),
        np.array([8, 9, 10, 11], dtype=np.float32),
    )
    assert reader.read_obs(2, columns=["cell_type"]) == {"cell_type": "c"}
