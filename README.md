# cellkit

`cellkit` is a small opinionated Python toolkit for building cell-model training repos on top of
AnnData and PyTorch.

It is intended to hold shared infrastructure that is generic across projects:

- lazy AnnData readers and datasets
- common preprocessing helpers
- transformer building blocks
- masking and masked-loss utilities
- run/output helpers such as checkpointing and async prediction logging

It is not a full training framework. Project-specific model wiring, collators, losses beyond the
shared primitives, and experiment scripts should usually live in the downstream repo.

## Installation

From the repo root:

```bash
uv sync
```

If another local repo uses `cellkit`, add it as an editable path dependency.

## What Lives Here

### `cellkit.data`

Utilities for working with disk-backed AnnData datasets.

- [src/cellkit/data/reader.py](/Users/feargusogorman/workspace/medai/FeargusOG/cellkit/src/cellkit/data/reader.py)
  Lazy readers for `.h5ad` and `.zarr`, plus `make_reader_factory(...)`.
- [src/cellkit/data/dataset.py](/Users/feargusogorman/workspace/medai/FeargusOG/cellkit/src/cellkit/data/dataset.py)
  A map-style PyTorch dataset that reads one observation at a time.
- [src/cellkit/data/preprocessing.py](/Users/feargusogorman/workspace/medai/FeargusOG/cellkit/src/cellkit/data/preprocessing.py)
  Generic preprocessing helpers: read/write AnnData, feature filtering, dropping empty cells,
  normalization, `log1p`, binning, and preprocessing metadata annotation.
- [src/cellkit/data/util.py](/Users/feargusogorman/workspace/medai/FeargusOG/cellkit/src/cellkit/data/util.py)
  Train/eval splitting, AnnData subsampling, and basic cell/gene filtering.
- [src/cellkit/data/explore.py](/Users/feargusogorman/workspace/medai/FeargusOG/cellkit/src/cellkit/data/explore.py)
  Lightweight dataset summaries and inspection helpers.

### `cellkit.model`

Reusable neural-network building blocks rather than one monolithic model class.

- embeddings
- MLP blocks
- normalization layers
- transformer layers
- simple heads

These are intended to be assembled by downstream repos into task-specific models.

### `cellkit.masking`

Shared masking logic for masked prediction tasks.

Current strategies:

- `uniform`
- `nonzero_only`
- `nonzero_preferred`

The main entrypoint is `build_prediction_mask(...)`, which supports a minimum masked-position
guarantee.

### `cellkit.losses`

Shared masked regression losses:

- `compute_masked_mse_loss(...)`
- `compute_masked_mae_loss(...)`

These operate on `pred`, `target`, and boolean `mask` tensors.

### `cellkit.utils`

Run-management helpers that are useful across training repos.

- [src/cellkit/utils/config.py](/Users/feargusogorman/workspace/medai/FeargusOG/cellkit/src/cellkit/utils/config.py)
  Experiment hashing, run directory creation, plain JSON artifact writing, file hashing, and git
  commit capture.
- [src/cellkit/utils/checkpoints.py](/Users/feargusogorman/workspace/medai/FeargusOG/cellkit/src/cellkit/utils/checkpoints.py)
  Opinionated `last.pt` / `best.pt` checkpoint saving plus `metadata.json`.
- [src/cellkit/utils/prediction_logging.py](/Users/feargusogorman/workspace/medai/FeargusOG/cellkit/src/cellkit/utils/prediction_logging.py)
  Best-effort async JSONL prediction logging with a bounded queue and background writer.

## Design Notes

The package is intentionally opinionated.

- AnnData is the main dataset abstraction.
- PyTorch is the model/training backend.
- Checkpointing currently assumes `accelerate.Accelerator`.
- Shared code should be generic enough to reuse across projects, but it does not need to be
  maximally abstract.

The rough rule is:

- if something is generic infrastructure used across multiple repos, it belongs in `cellkit`
- if something defines one project's exact task semantics, it should stay in that project

## Example Usage

### Build a Reader-Backed Dataset

```python
from cellkit.data.dataset import AnnDataDataset
from cellkit.data.reader import make_reader_factory

reader_factory = make_reader_factory("data/example.zarr", "zarr")
dataset = AnnDataDataset(reader_factory, return_index=True)
sample = dataset[0]
```

### Preprocess an AnnData Object

```python
from cellkit.data.preprocessing import (
    filter_genes,
    drop_cells_with_no_expression,
    normalize_total,
    log1p_transform,
)

adata = filter_genes(adata, genes_to_keep)
adata = drop_cells_with_no_expression(adata)
adata = normalize_total(adata, target_sum=10_000)
adata = log1p_transform(adata)
```

### Save Shared Run Artifacts

```python
from cellkit.utils.config import setup_run_dirs, write_run_artifacts

out_paths = setup_run_dirs(train_config, output_dir="./out", run_title="demo")
write_run_artifacts(
    run_dir=out_paths["run_dir"],
    run_config=run_config,
    flat_train_config=flat_train_config,
    manifest=manifest,
)
```

## Scripts

The repo also contains a few small utility scripts:

- [scripts/explore_dataset.py](/Users/feargusogorman/workspace/medai/FeargusOG/cellkit/scripts/explore_dataset.py)
  Inspect an AnnData dataset.
- [scripts/subsample_dataset.py](/Users/feargusogorman/workspace/medai/FeargusOG/cellkit/scripts/subsample_dataset.py)
  Create a smaller stratified subset.

Example:

```bash
uv run python scripts/explore_dataset.py --adata data/CD45.h5ad --rows cell_subset cell_subtype_3
uv run python scripts/subsample_dataset.py --adata data/CD45.h5ad --out data/CD45.subsampled.h5ad --frac 0.1 --strata-cols cell_subtype_3
```

## Development

Run the test suite from the `cellkit` repo root:

```bash
uv run pytest tests
```

The tests are organized by module area:

- `tests/data`
- `tests/model`
- `tests/utils`
- top-level tests for shared modules like masking and losses

## Current Scope

`cellkit` is currently best thought of as shared infrastructure for personal downstream projects,
not a polished public package with a stable API. Module boundaries are meaningful, but interfaces may
still evolve as more projects adopt them.
