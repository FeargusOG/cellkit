from sklearn.model_selection import train_test_split
import numpy as np


def stratified_subsample_adata(
    adata,
    frac: float,
    strata_cols: list | None = None,
    random_state: int = 0,
    drop_small_strata: bool = True,
):
    """
    Subsample AnnData to retain a given fraction of cells, optionally stratifying by strata_cols.

    Args:
        adata (AnnData): Input AnnData.
        frac (float): Fraction of cells to keep (0 < frac <= 1).
        strata_cols (list or None): List of `obs` column names to balance across.
        random_state (int): Random seed for reproducibility.
        drop_small_strata (bool): If True, drop strata with fewer than 2 cells.

    Returns:
        AnnData: Subsampled AnnData object.
    """
    if frac <= 0 or frac > 1:
        raise ValueError("frac must be in the interval (0, 1].")

    obs_df = adata.obs.copy()
    obs_df["__index"] = obs_df.index

    if strata_cols:
        stratify_vals = obs_df[strata_cols].astype(str).agg("_".join, axis=1)

        # train_test_split will error with singleton groups!
        group_counts = stratify_vals.value_counts()
        if (group_counts < 2).any():
            if drop_small_strata:
                valid_groups = group_counts[group_counts >= 2].index
                mask = stratify_vals.isin(valid_groups)
                obs_df = obs_df[mask]
                stratify_vals = stratify_vals[mask]
            else:
                print(
                    "Warning: Some strata have fewer than 2 cells. "
                    "Falling back to unstratified sampling. "
                    "Set drop_small_strata=True to drop small strata."
                )
                stratify_vals = None
    else:
        stratify_vals = None

    _, subsample_idx = train_test_split(
        obs_df["__index"],
        stratify=stratify_vals,
        test_size=frac,
        random_state=random_state,
    )

    return adata[subsample_idx].copy()
