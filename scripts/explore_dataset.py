import anndata as ad
import argparse
import numpy as np
from scipy.sparse import issparse
from cellkit.data import explore

parser = argparse.ArgumentParser()
parser.add_argument("--adata", required=True, help="Path to the .h5ad or .zarr file")
parser.add_argument(
    "--rows", nargs="+", default=None, help="List of rows to print more details for"
)
args = parser.parse_args()

adata = ad.read_h5ad(args.adata)

print("\n")
print("********************************")
print("*         General Info         *")
print("********************************")
print("\n")
print("(cells, genes)")
print(adata.shape)
print("\n")
print(f"X is Sparse: {issparse(adata.X)}")
print(f"Type of X: {type(adata.X)}")

print("\n")
print("********************************")
print("*      Cell and Gene Info      *")
print("********************************")
explore.summarize_expression_coverage(adata)

print("\n")
print("********************************")
print("*                              *")
print("*         Observations         *")
print("*      aka Rows aka Cells      *")
print("*                              *")
print("********************************")

print("\n")
print("\t *******************************")
print("\t **         Head(n=3)         **")
print("\t *******************************")
print("\n")
print(adata.obs.head(3).T)  # type: ignore[attr-defined]

rows = args.rows
if rows != None:
    for row in args.rows:
        print("\n")
        print("\t *******************************")
        print(f"\t **    Counts: {row}    **")
        print("\t *******************************")
        print("\n")
        print(adata.obs[row].value_counts())

print("\n")
print("********************************")
print("*                              *")
print("*           Variables          *")
print("*      aka Cols aka Genes      *")
print("*                              *")
print("********************************")

print("\n")
print("\t *******************************")
print("\t **         Head(n=3)         **")
print("\t *******************************")
print("\n")
print(adata.var.head(n=3).T)  # type: ignore[attr-defined]

print("\n")
print("********************************")
print("*                              *")
print("*            Layers            *")
print("*                              *")
print("********************************")

if len(adata.layers) == 0:
    print("\nNo layers found in this AnnData object.")
else:
    print("\nAvailable layers:", list(adata.layers.keys()))
    for layer_name in adata.layers:
        layer_data = adata.layers[layer_name]
        from scipy.sparse import issparse

        print(f"\nLayer: {layer_name}")
        print(f"  Shape: {layer_data.shape}")
        print(f"  Sparse: {issparse(layer_data)}")
        print(f"  Type: {type(layer_data)}")

print("\n")
print("********************************")
print("*                              *")
print("*       X Value Summary        *")
print("*                              *")
print("********************************")

if issparse(adata.X):
    X = adata.X.tocoo()  # type: ignore[attr-defined]
    print(f"  Non-zero entries: {X.nnz}")
    print(f"  Min (non-zero): {X.data.min()}")
    print(f"  Max (non-zero): {X.data.max()}")
    print(f"  Mean (non-zero): {X.data.mean():.3f}")
    print(f"  Median (non-zero): {np.median(X.data):.3f}")  # type: ignore[attr-defined]
else:
    print(f"  Min: {adata.X.min()}")  # type: ignore[attr-defined]
    print(f"  Max: {adata.X.max()}")  # type: ignore[attr-defined]
    print(f"  Mean: {adata.X.mean():.3f}")  # type: ignore[attr-defined]
    print(f"  Median: {np.median(adata.X):.3f}")  # type: ignore[]
