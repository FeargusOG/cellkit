import anndata as ad


def read_anndata(path):
    if path.endswith(".zarr"):
        return ad.read_zarr(path)
    if path.endswith(".h5ad"):
        return ad.read_h5ad(path)
    return ad.read(path)
