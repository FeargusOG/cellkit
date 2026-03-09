from unittest import mock
import pytest

from cellkit.data.io import read_anndata


def test_read_anndata_routes_by_extension():
    with mock.patch("cellkit.data.io.ad.read_h5ad") as read_h5ad, mock.patch(
        "cellkit.data.io.ad.read_zarr"
    ) as read_zarr:
        read_h5ad.return_value = "h5ad"
        read_zarr.return_value = "zarr"

        assert read_anndata("file.h5ad") == "h5ad"
        read_h5ad.assert_called_once_with("file.h5ad")

        assert read_anndata("file.zarr") == "zarr"
        read_zarr.assert_called_once_with("file.zarr")

        with pytest.raises(ValueError, match="Unsupported AnnData format"):
            read_anndata("file.txt")
