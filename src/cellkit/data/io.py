import anndata as ad


def read_anndata(path):
    path_str = str(path)
    if path_str.endswith(".zarr"):
        return ad.read_zarr(path)
    if path_str.endswith(".h5ad"):
        return ad.read_h5ad(path)
    if hasattr(ad, "read"):
        return ad.read(path)
    raise ValueError("Unsupported AnnData format; expected .h5ad or .zarr")


def write_anndata(adata, path):
    path_str = str(path)
    if path_str.endswith(".zarr"):
        adata.write_zarr(path)
    elif path_str.endswith(".h5ad"):
        adata.write_h5ad(path)
    else:
        adata.write(path)
