import argparse

from cellkit.data import io, util


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

adata = io.read_anndata(args.adata)
sub = util.stratified_subsample_adata(
    adata,
    frac=args.frac,
    strata_cols=args.strata_cols,
    random_state=args.random_state,
    drop_small_strata=args.drop_small_strata,
)
io.write_anndata(sub, args.out)

print(f"Wrote subsampled AnnData to {args.out}")
print(f"Original cells: {adata.n_obs}")
print(f"Subsampled cells: {sub.n_obs}")
