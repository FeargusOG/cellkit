from pathlib import Path
import pickle
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from cellkit.data.reader import H5ADReader
from cellkit.data.reader import ZarrReader
from cellkit.data.reader import open_reader


class FakeAnnData:
    def __init__(self):
        self.X = np.arange(12, dtype=np.float32).reshape(3, 4)
        self.layers = {
            "counts": sparse.csr_matrix(np.arange(12, dtype=np.float32).reshape(3, 4))
        }
        self.obs = pd.DataFrame(
            {"cell_type": ["a", "b", "c"], "batch": [0, 1, 1]},
            index=["cell0", "cell1", "cell2"],
        )
        self.n_obs = 3
        self.n_vars = 4
        self.file = mock.Mock()


def test_open_reader_routes_by_extension():
    assert isinstance(open_reader("dataset.h5ad"), H5ADReader)
    assert isinstance(open_reader(Path("dataset.zarr")), ZarrReader)

    with pytest.raises(ValueError, match="Unsupported dataset format"):
        open_reader("dataset.txt")


def test_h5ad_reader_uses_backed_mode_and_reads_rows():
    fake_adata = FakeAnnData()
    with mock.patch("cellkit.data.reader.ad.read_h5ad", return_value=fake_adata) as read:
        reader = H5ADReader("dataset.h5ad")

        assert len(reader) == 3
        assert reader.shape == (3, 4)
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

        np.testing.assert_array_equal(reader.read_x(0), np.array([0, 1, 2, 3]))
        assert reader.read_obs(2, columns=["batch"]) == {"batch": 1}

        read_lazy.assert_called_once_with(Path("dataset.zarr"))


def test_reader_pickling_drops_open_anndata_handle():
    fake_adata = FakeAnnData()
    with mock.patch("cellkit.data.reader.ad.read_h5ad", return_value=fake_adata):
        reader = H5ADReader("dataset.h5ad")
        assert len(reader) == 3

        restored = pickle.loads(pickle.dumps(reader))

        fake_adata.file.close.assert_not_called()
        assert reader._adata is fake_adata
        assert restored._adata is None
