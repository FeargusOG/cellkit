import argparse
from scipy.sparse import issparse
from cellkit.data import explore, io

parser = argparse.ArgumentParser()
parser.add_argument("--adata", required=True, help="Path to the .h5ad or .zarr file")
parser.add_argument(
    "--layers",
    nargs="+",
    default=["X"],
    help="One or more layers to summarize (use X for adata.X)",
)
parser.add_argument(
    "--rows", nargs="+", default=None, help="List of rows to print more details for"
)
args = parser.parse_args()

adata = io.read_anndata(args.adata)

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
for layer_name in args.layers:
    explore.summarize_expression_coverage(adata, layer=layer_name)

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
if rows is not None:
    missing_rows = [row for row in rows if row not in adata.obs.columns]
    if missing_rows:
        print("\n")
        print(f"Warning: Missing obs columns: {missing_rows}")
    for row in rows:
        if row not in adata.obs.columns:
            continue
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

        print(f"\nLayer: {layer_name}")
        print(f"  Shape: {layer_data.shape}")
        print(f"  Sparse: {issparse(layer_data)}")
        print(f"  Type: {type(layer_data)}")

print("\n")
print("********************************")
print("*                              *")
print("*        Value Summary         *")
print("*                              *")
print("********************************")

for layer_name in args.layers:
    explore.summarize_value_stats(adata, layer=layer_name)
