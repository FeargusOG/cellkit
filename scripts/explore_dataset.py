import anndata as ad
from scipy.sparse import issparse

from cellkit.data import explore


adata = ad.read_h5ad("data/CD45.h5ad")

print("\n")
print("********************************")
print("*        General Adata         *")
print("********************************")
print("\n")
print("(cells, genes)")
print(adata.shape)
print("\n")
print(f"X is Sparse: {issparse(adata.X)}")
print(f"Type of X: {type(adata.X)}")

explore.summarize_expression_coverage(adata)
