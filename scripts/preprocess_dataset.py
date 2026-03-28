import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from cellkit.data.preprocessing import preprocess_anndata
from cellkit.data.preprocessing import read_anndata
from cellkit.data.preprocessing import write_anndata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", required=True, help="Input .h5ad or .zarr path")
    parser.add_argument("--output-data", required=True, help="Output .h5ad or .zarr path")
    parser.add_argument(
        "--genes-path",
        default=None,
        help="Optional path to a JSON list/dict or newline-delimited text file of genes to keep",
    )
    parser.add_argument(
        "--target-sum",
        type=float,
        default=None,
        help="If set, normalize each cell to this total count before any optional log1p/binning",
    )
    parser.add_argument(
        "--log1p",
        action="store_true",
        help="Apply log1p after optional total-count normalization",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=None,
        help="If set, bin nonzero expression values per cell into 1..n_bins while keeping zeros at 0",
    )
    parser.add_argument(
        "--keep-zero-expression-cells",
        action="store_true",
        help="Keep cells that have zero total expression after any optional filtering",
    )
    return parser.parse_args()


def load_gene_set(path: str | None) -> set[str] | None:
    if path is None:
        return None

    gene_path = Path(path)
    text = gene_path.read_text().strip()
    if not text:
        raise ValueError("genes-path file is empty")

    if gene_path.suffix == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            return {str(key) for key in payload.keys()}
        if isinstance(payload, list):
            return {str(item) for item in payload}
        raise ValueError("genes-path JSON must be a list or object")

    return {line.strip() for line in text.splitlines() if line.strip()}


def main() -> None:
    args = parse_args()
    adata = read_anndata(args.input_data)
    genes_to_keep = load_gene_set(args.genes_path)

    adata = preprocess_anndata(
        adata,
        genes_to_keep=genes_to_keep,
        drop_zero_expression_cells=not args.keep_zero_expression_cells,
        target_sum=args.target_sum,
        log1p=args.log1p,
        n_bins=args.n_bins,
    )
    write_anndata(adata, args.output_data)

    print(f"Wrote preprocessed AnnData to {args.output_data}")
    print(f"Cells: {adata.n_obs}")
    print(f"Genes: {adata.n_vars}")


if __name__ == "__main__":
    main()
