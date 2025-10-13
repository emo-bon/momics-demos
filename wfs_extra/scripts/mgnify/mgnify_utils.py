"""
Reusable utilities extracted from the MGnify notebooks.
"""
from __future__ import annotations

import os
import json
import pandas as pd
from typing import Iterable, Optional, Sequence, Tuple, Dict, Any

from jsonapi_client import Session as APISession


# ---------------------------
# Download helpers
# ---------------------------

def retrieve_summary(studyId: str, matching_string: str = 'Taxonomic assignments SSU', out_dir: str = '.') -> str:
    """
    Retrieve summary data for a given MGnify study and save it to a TSV file.

    Parameters
    ----------
    studyId : str
        MGnify study ID, e.g., 'MGYS00006680'.
    matching_string : str
        Label to match in download description (e.g., 'Taxonomic assignments SSU').
    out_dir : str
        Output directory to store the TSV file.

    Returns
    -------
    str
        Path to the downloaded TSV file.
    """
    from urllib.request import urlretrieve

    os.makedirs(out_dir, exist_ok=True)
    tsv_path = os.path.join(out_dir, f'{studyId}.tsv')

    with APISession("https://www.ebi.ac.uk/metagenomics/api/v1") as session:
        for download in session.iterate(f"studies/{studyId}/downloads"):
            if download.description.label == matching_string:
                urlretrieve(download.links.self.url, tsv_path)
                return tsv_path

    raise FileNotFoundError(f"No download matched '{matching_string}' for study {studyId}")


# ---------------------------
# Taxonomy utilities
# ---------------------------

def pivot_taxonomic_data(
    df: pd.DataFrame,
    abundance_col: str = "abundance",
    tax_id_col: str = "ncbi_tax_id",
    taxonomy_ranks=None,
    concat_col: str = "taxonomic_concat",
    drop_missing_tax_id: bool = False,
    fill_missing: bool = True,
    strict: bool = False,
) -> pd.DataFrame:
    """Create a pivoted abundance matrix from long-form taxonomic data.

    This is the refactored version from the notebook.
    """
    if taxonomy_ranks is None:
        taxonomy_ranks = [
            "superkingdom",
            "kingdom",
            "phylum",
            "class",
            "order",
            "family",
            "genus",
            "species",
        ]

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if abundance_col not in df.columns:
        raise KeyError(f"Missing required abundance column '{abundance_col}'")

    df_work = df.copy()

    if isinstance(df_work.index, pd.MultiIndex):
        first_level_name = df_work.index.names[0] or "sample"
        df_work = df_work.reset_index(level=list(range(df_work.index.nlevels)))
        df_work = df_work.set_index(first_level_name)

    has_tax_id = tax_id_col in df_work.columns
    if strict and not has_tax_id:
        raise KeyError(f"Required taxonomic id column '{tax_id_col}' not found and strict=True")

    if drop_missing_tax_id and has_tax_id:
        df_work = df_work[~df_work[tax_id_col].isna()].copy()

    def _fill(series: pd.Series) -> pd.Series:
        return series.fillna("") if fill_missing else series

    parts = []
    if has_tax_id:
        parts.append(df_work[tax_id_col].astype(str))

    for rank in taxonomy_ranks:
        if rank not in df_work.columns:
            series_part = pd.Series(["" if fill_missing else None] * len(df_work), index=df_work.index)
        else:
            series_part = _fill(df_work[rank]).astype(str)
        prefix = "sk__" if rank == "superkingdom" else f"{rank[0]}__"
        parts.append(prefix + series_part)

    df_work[concat_col] = parts[0] if parts else ""
    if has_tax_id:
        df_work[concat_col] = df_work[concat_col] + ";"
        start_index = 1
    else:
        df_work[concat_col] = ""
        start_index = 0

    for p in parts[start_index:]:
        df_work[concat_col] = df_work[concat_col] + p.fillna("") + ";"

    df_work[abundance_col] = pd.to_numeric(df_work[abundance_col], errors="coerce").fillna(0).astype(int)

    pivot_index = [concat_col] if not has_tax_id else [tax_id_col, concat_col]

    pivot = (
        df_work.pivot_table(
            index=pivot_index,
            columns=df_work.index,
            values=abundance_col,
            aggfunc="sum",
            fill_value=0,
        )
        .astype(int)
        .sort_index()
    )

    return pivot


def invert_pivot_taxonomic_data(
    pivot: pd.DataFrame,
    drop_zeros: bool = True,
    target_col: str = "taxonomic_concat",
) -> pd.DataFrame:
    """Invert pivoted taxonomic data back to long form (simplified)."""
    if not isinstance(pivot, pd.DataFrame):
        raise TypeError("pivot must be a pandas DataFrame")

    reset = pivot.reset_index()
    melted = reset.melt(
        id_vars=[c for c in [target_col] if c in reset.columns],
        value_vars=[c for c in reset.columns if c not in [target_col]],
        var_name='sample',
        value_name="abundance",
    )

    if drop_zeros:
        melted = melted[melted["abundance"] != 0].copy()

    try:
        melted["abundance"] = melted["abundance"].astype(int)
    except Exception:
        melted["abundance"] = pd.to_numeric(melted["abundance"], errors="coerce")

    return melted


# ---------------------------
# IO helpers
# ---------------------------

def save_config(config: Dict[str, Any], out_dir: str, filename: str = "config.json") -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, sort_keys=True)
    return path


# ---------------------------
# Taxonomy parsing helpers
# ---------------------------

_PREFIX_MAP = {
    "superkingdom": "sk__",
    "kingdom": "k__",
    "phylum": "p__",
    "class": "c__",
    "order": "o__",
    "family": "f__",
    "genus": "g__",
    "species": "s__",
}


def parse_taxonomic_concat(value: str, taxonomy_ranks: Sequence[str] | None = None) -> Dict[str, Any]:
    """Parse a semicolon-separated taxonomic_concat string into rank columns.

    Example of expected pattern:
        "sk__Bacteria;k__...;p__Firmicutes;...;g__Lactobacillus;s__acidophilus"

    Returns a dict mapping rank name -> value (empty string if missing).
    """
    if taxonomy_ranks is None:
        taxonomy_ranks = list(_PREFIX_MAP.keys())

    result: Dict[str, Any] = {r: "" for r in taxonomy_ranks}
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return result

    parts = [p.strip() for p in str(value).split(";") if p.strip()]
    for part in parts:
        for rank in taxonomy_ranks:
            pref = _PREFIX_MAP.get(rank, f"{rank[0]}__")
            if part.startswith(pref):
                result[rank] = part[len(pref):].strip()
                break
    return result


def wide_to_long_with_ranks(
    df_wide: pd.DataFrame,
    taxonomy_col: str = "taxonomy",
    abundance_col: str = "abundance",
    taxonomy_ranks: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Convert wide table (taxonomy rows x sample columns) to long with rank columns.

    Parameters
    ----------
    df_wide : DataFrame with taxonomy strings in `taxonomy_col` and samples as columns.
    taxonomy_col : Name of the column/index holding taxonomy string (e.g., 'taxonomy').
    abundance_col : Name for the abundance value column in the long output.
    taxonomy_ranks : Optional list of ranks to extract; defaults to common ranks.

    Returns
    -------
    DataFrame with columns: sample, abundance, and one column per taxonomy rank.
    """
    if taxonomy_ranks is None:
        taxonomy_ranks = list(_PREFIX_MAP.keys())

    if df_wide.index.name == taxonomy_col or taxonomy_col in getattr(df_wide, "columns", []):
        df2 = df_wide.copy()
    else:
        # Try to reset index to find taxonomy column
        df2 = df_wide.copy()
        df2.index.name = taxonomy_col

    if taxonomy_col not in df2.columns:
        df2 = df2.reset_index()

    long_df = df2.melt(
        id_vars=[taxonomy_col],
        var_name="sample",
        value_name=abundance_col,
    )

    # Drop zeros quickly to reduce parsing cost
    long_df = long_df[long_df[abundance_col] != 0].copy()

    parsed = long_df[taxonomy_col].apply(lambda v: parse_taxonomic_concat(v, taxonomy_ranks))
    parsed_df = pd.DataFrame(parsed.tolist(), index=long_df.index)
    out = pd.concat([long_df.drop(columns=[taxonomy_col]), parsed_df], axis=1)

    # Ensure abundance numeric int
    out[abundance_col] = pd.to_numeric(out[abundance_col], errors="coerce").fillna(0).astype(int)
    return out
