# Cell Kit

## Scripts

### Explore Dataset

- Example usage: `uv run python scripts/explore_dataset.py --adata data/CD45.h5ad --rows cell_subset cell_subtype_3`

### Subsample Dataset

- Example usage: `uv run python scripts/subsample_dataset.py --adata data/CD45.h5ad --out data/CD45.subsampled.h5ad --frac 0.1 --strata-cols cell_subtype_3`
