import anndata as ad


def read_anndata(path):
    if path.endswith(".zarr"):
        return ad.read_zarr(path)
    if path.endswith(".h5ad"):
        return ad.read_h5ad(path)
    if hasattr(ad, "read"):
        return ad.read(path)
    raise ValueError("Unsupported AnnData format; expected .h5ad or .zarr")
