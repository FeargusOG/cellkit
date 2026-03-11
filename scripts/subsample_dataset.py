import argparse
import sys
from pathlib import Path
import anndata as ad

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from cellkit.data import util

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


parser = argparse.ArgumentParser()
parser.add_argument("--adata", required=True, help="Path to the .h5ad or .zarr file")
parser.add_argument("--out", required=True, help="Output path (.h5ad or .zarr)")
parser.add_argument(
    "--frac",
    required=True,
    type=float,
    help="Fraction of cells to keep (0 < frac <= 1)",
)
parser.add_argument(
    "--strata-cols",
    nargs="+",
    default=None,
    help="List of obs column names to stratify by",
)
parser.add_argument("--random-state", type=int, default=0, help="Random seed")
parser.add_argument(
    "--drop-small-strata",
    action="store_true",
    help="Drop strata with fewer than 2 cells",
)
args = parser.parse_args()

adata = read_anndata(args.adata)
sub = util.stratified_subsample_adata(
    adata,
    frac=args.frac,
    strata_cols=args.strata_cols,
    random_state=args.random_state,
    drop_small_strata=args.drop_small_strata,
)
write_anndata(sub, args.out)

print(f"Wrote subsampled AnnData to {args.out}")
print(f"Original cells: {adata.n_obs}")
print(f"Subsampled cells: {sub.n_obs}")
