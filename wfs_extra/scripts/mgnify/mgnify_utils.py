"""
Reusable utilities extracted from the MGnify notebooks.
"""
from __future__ import annotations

import os
import json
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import sys
from typing import Iterable, Optional, Sequence, Tuple, Dict, Any
from tqdm import tqdm

from jsonapi_client import Session as APISession


# ---------------------------
# Download helpers
# ---------------------------
def retrieve_summary_old(studyId: str, matching_string: str = 'Taxonomic assignments SSU') -> None:
    """
    Retrieve summary data for a given analysis ID and save it to a file. Matching strings 
    are substrings of for instance:
    - Phylum level taxonomies SSU
    - Taxonomic assignments SSU
    - Phylum level taxonomies LSU
    - Taxonomic assignments LSU
    - Phylum level taxonomies ITSoneDB
    - Taxonomic assignments ITSoneDB
    - Phylum level taxonomies UNITE
    - Taxonomic assignments UNITE

    Example usage:
    retrieve_summary('MGYS00006680', matching_string='Taxonomic assignments SSU')

    Args:
        studyId (str): The ID of the analysis to retrieve. studyId is the MGnify study ID, used
            also to save the output .tsv file.
        matching_string (str): The string to match in the download description label.
    
    Returns:
        None
    """
    from urllib.request import urlretrieve

    with APISession("https://www.ebi.ac.uk/metagenomics/api/v1") as session:
        for download in session.iterate(f"studies/{studyId}/downloads"):
            if download.description.label == matching_string:
                print(f"Downloading {download.alias}...")
                urlretrieve(download.links.self.url, f'{studyId}.tsv')

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

# function to get metadata for MGnify studies
def get_mgnify_metadata(study_id):
    with APISession("https://www.ebi.ac.uk/metagenomics/api/v1") as session:

        samples = map(lambda r: r.json, session.iterate(f'studies/{study_id}/samples?page_size=1000'))

        sample_list = []
        for sample_json in tqdm(samples):
            # Flatten sample-metadata list into a dictionary
            # 1. Extract sample-metadata (allowing None)
            metadata_fields = {
                item.get("key"): item.get("value", None)
                for item in sample_json["attributes"].get("sample-metadata", [])
            }

            # 2. Extract all other attributes (including None)
            attributes_fields = {
                k: v for k, v in sample_json["attributes"].items()
                if k != "sample-metadata"  # already unpacked separately
            }

            # 3. Merge everything including top-level id
            flat_data = {
                "id": sample_json.get("id"),
                **attributes_fields,
                **metadata_fields
            }

            # 4. Create DataFrame
            df = pd.DataFrame([flat_data])
            sample_list.append(df)

        # Concatenate all DataFrames into one
        df = pd.concat(sample_list, ignore_index=True)
        df['study'] = study_id
    return df

# ---------------------------
# Taxonomy utilities
# ---------------------------
def pivot_taxonomic_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the taxonomic data (LSU and SSU tables) for analysis. Apart from
    pivoting.

    Normalization of the pivot is optional. Methods include:

    - **None**: no normalization.
    - **tss_sqrt**: Total Sum Scaling and Square Root Transformation.
    - **rarefy**: rarefaction to a specified depth, if None, min of sample sums is used.

    TODO: refactor scaling to a new method and offer different options.

    Args:
        df (pd.DataFrame): The input DataFrame containing taxonomic information.
        normalize (str, optional): Normalization method.
            Options: None, 'tss_sqrt', 'rarefy'. Defaults to None.
        rarefy_depth (int, optional): Depth for rarefaction. If None, uses min sample sum.
            Defaults to None.

    Returns:
        pd.DataFrame: A pivot table with taxonomic data.
    """
    if isinstance(df.index, pd.MultiIndex):
        index = df.index.names[0]
        df1 = df.reset_index()
        df1.set_index(index, inplace=True)
    else:
        df1 = df.copy()

    tax_ranks = [
        'ncbi_tax_id', 'superkingdom', 'kingdom', 'phylum', 'class',
        'order', 'family', 'genus', 'species',
    ]
    prefix_map = {
        "ncbi_tax_id": "",
        "superkingdom": "sk__",
        "kingdom": "k__",
        "phylum": "p__",
        "class": "c__",
        "order": "o__",
        "family": "f__",
        "genus": "g__",
        "species": "s__",
    }
    tax_ranks_filt = [tax for tax in tax_ranks if tax in df1.columns]
    df1["taxonomic_concat"] = df1.apply(
        lambda row: ";" + ";".join(
            f"{prefix_map[tax]}{row[tax]}" if pd.notna(row[tax]) else f"{prefix_map[tax]}"
            for tax in tax_ranks_filt
        ),
        axis=1
    )
    if 'ncbi_tax_id' in tax_ranks_filt:
        pivot_table = (
            df1.pivot_table(
                index=["ncbi_tax_id", "taxonomic_concat"],
                columns=df1.index,
                values="abundance",
            )
            .fillna(0)
            .astype(int)
        )
    else:
        pivot_table = (
            df1.pivot_table(
                index=["taxonomic_concat"],
                columns=df1.index,
                values="abundance",
            )
            .fillna(0)
            .astype(int)
        )
    return pivot_table


def pivot_taxonomic_data_new(
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


def invert_pivot_taxonomic_data_new(
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


def invert_pivot_taxonomic_data(
    pivot: pd.DataFrame,
    drop_zeros: bool = True,
    target_col: str = "taxonomic_concat",
) -> pd.DataFrame:
    """
    Invert the pivot table produced by `pivot_taxonomic_data` back to a long-form table.

    Args:
        pivot (pd.DataFrame): Pivot table with index=['ncbi_tax_id', 'taxonomic_concat']
                             and columns = original sample ids (index values).
        drop_zeros (bool): If True, rows with abundance == 0 are removed. Default True.

    Returns:
        pd.DataFrame: Long-form DataFrame with columns:
                      [sample_col_name, 'ncbi_tax_id', 'taxonomic_concat',
                       'superkingdom','kingdom','phylum','class','order','family','genus','species',
                       'abundance']
    """
    # input checks
    if not isinstance(pivot, pd.DataFrame):
        raise TypeError("pivot must be a pandas DataFrame")


    # Reset index so we have ncbi_tax_id and taxonomic_concat as columns
    reset = pivot.reset_index()
    # reset = pivot.copy()
    # Melt wide -> long
    melted = reset.melt(
        id_vars=[target_col],
        value_vars=[c for c in reset.columns if c != target_col],
        var_name='sample',
        value_name="abundance",
    )

    # Optionally drop zeros
    if drop_zeros:
        melted = melted[melted["abundance"] != 0].copy()

    # Ensure abundance integer type where possible
    try:
        melted["abundance"] = melted["abundance"].astype(int)
    except Exception:
        # fallback to numeric if some values are non-integer
        melted["abundance"] = pd.to_numeric(melted["abundance"], errors="coerce")

    # Parse taxonomic_concat into components.
    # Expected pattern (example):
    # "12345;sk__Archaea;k__...;p__Phylum;c__Class;...;g__Genus;s__Species"
    tax_columns = [
        ("superkingdom", r"sk__"),
        ("kingdom", r"k__"),
        ("phylum", r"p__"),
        ("class", r"c__"),
        ("order", r"o__"),
        ("family", r"f__"),
        ("genus", r"g__"),
        ("species", r"s__"),
    ]

    def parse_tax_concat(s: str):
        # initialize result with empty strings
        res = {col: "" for col, _ in tax_columns}
        if pd.isna(s):
            return res
        # split by ';'
        parts = [p.strip() for p in s.split(";") if p.strip() != ""]
        # first part may be ncbi_tax_id (numeric) - but we rely on explicit ncbi_tax_id column already
        for p in parts:
            for col, prefix in tax_columns:
                if p.startswith(prefix):
                    # remove prefix and use remainder; keep empty if nothing after prefix
                    value = p[len(prefix) :].strip()
                    # normalize empty strings to NaN? Keep as empty string for now
                    res[col] = value
                    break
        return res

    # Apply parser and expand into separate columns
    parsed = melted[target_col].apply(parse_tax_concat)
    parsed_df = pd.DataFrame(parsed.tolist(), index=melted.index)

    result = pd.concat([melted.reset_index(drop=True), parsed_df.reset_index(drop=True)], axis=1)

    # Reorder columns for readability
    cols_order = [
        "sample",
        "abundance",
        "superkingdom",
        "kingdom",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
    ]
    # keep only columns that exist (in case some are missing)
    cols_order = [c for c in cols_order if c in result.columns]
    result = result[cols_order]

    # Reset index to a clean integer index
    result = result.reset_index(drop=True)

    return result

def fill_lower_taxa(df: pd.DataFrame, taxonomy_ranks: list) -> pd.DataFrame:
    """
    Fill lower taxonomy ranks with None if the current rank is empty and the lower rank is also empty.
    Starts with the lowest rank and moves upwards.
    
    Args:
        df (pd.DataFrame): DataFrame with taxonomic ranks as columns.
        taxonomy_ranks (list): List of taxonomy rank column names in hierarchical order.

    Returns:
        pd.DataFrame: DataFrame with lower taxonomy ranks filled with None where appropriate.
    """
    df_out = df.copy()
    df_out[taxonomy_ranks[-1]] = df_out[taxonomy_ranks[-1]].replace('', None)
    for i in range(2, len(taxonomy_ranks)):
        lower = taxonomy_ranks[-i + 1]  # lower rank column
        current = taxonomy_ranks[-i]

        df_out[current] = df_out.apply(
        lambda row: None if (row[current] == '' and pd.isna(row[lower])) else row[current],
        axis=1
    )
    return df_out

def aggregate_by_taxonomic_level(df: pd.DataFrame, level: str, dropna: bool = True) -> pd.DataFrame:
    """
    Aggregates the DataFrame by a specific taxonomic level and sums abundances across samples.

    Args:
        df (pd.DataFrame): The input DataFrame containing taxonomic information.
        level (str): The taxonomic level to aggregate by (e.g., 'phylum', 'class', etc.).
        dropna (bool): If True, rows with NaN in the specified level and all higher levels
            are dropped before aggregation. Default is True.

    Returns:
        pd.DataFrame: A DataFrame aggregated by the specified taxonomic level.
    """
    TAXONOMY_RANKS = ['superkingdom', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    if level not in df.columns:
        raise KeyError(f"Taxonomic level '{level}' not found in DataFrame")
    
    levels = TAXONOMY_RANKS[:TAXONOMY_RANKS.index(level)+1]

    # working = df if not dropna else df.dropna(subset=[level])
    # working = df if not dropna else df.dropna(subset=levels)
    working = df.copy()

    # Group by the specified level and sum abundances across samples (columns)
    grouped = (
        working
        .groupby([working.index.name, *levels], dropna=dropna)
        .sum(numeric_only=True)
        .reset_index()
        .set_index(working.index.name)
    )
    return grouped

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


def prevalence_cutoff_abund(
    df: pd.DataFrame, percent: float = 10, skip_columns: int = 2, verbose: bool = True
) -> pd.DataFrame:
    """
    Apply a prevalence cutoff to the DataFrame, removing features that have abundance
    lower than `percent` in the sample. This goes sample (column) by sample independently.

    Args:
        df (pd.DataFrame): The input DataFrame containing feature abundances.
        percent (float): The prevalence threshold as a percentage.
        skip_columns (int): The number of columns to skip (e.g., taxonomic information).

    Returns:
        pd.DataFrame: A filtered DataFrame with low-prevalence features removed.
    """
    out = df.copy()
    max_threshold = 0
    for col in df.iloc[:, skip_columns:]:
        threshold = (percent / 100) * df[col].sum()

        # how many are below threshold?
        max_threshold = max(max_threshold, threshold)
        
        # set to zero those below threshold
        out.loc[df[col] < threshold, col] = 0


    # remove rows that are all zeros in the abundance columns
    out = out[(out.iloc[:, skip_columns:] != 0).any(axis=1)]
    if verbose:
        print(f"Prevalence cutoff at {percent}% (max threshold {max_threshold}): reduced from {df.shape} to {out.shape}")
    return out

# -----------------------
# Rarefaction curve
# -----------------------
def rarefaction_curve(reads, steps=20, replicates=10):
    depths = np.linspace(1, len(reads), steps, dtype=int)
    richness = []
    for n in depths:
        reps = []
        for _ in range(replicates):
            subsample = np.random.choice(reads, size=n, replace=False)
            reps.append(len(np.unique(subsample)))
        richness.append(np.mean(reps))
    return depths, richness


def plot_rarefaction_mgnify(abund_table, metadata, every_nth=20, ax=None, title="Rarefaction curves per sample"):
    if ax is None:
        fig, ax = plt.subplots()
    for sample in abund_table.columns[::every_nth]:
        _, ratio = extract_sample_stats(metadata, sample)
        reads = np.repeat(abund_table.index, abund_table[sample].values)
        depths, richness = rarefaction_curve(reads)

        ax.plot(depths, richness, label=f'{sample} (unidentified ratio: {ratio:.2f})')

    ax.legend()
    ax.set_xlabel("Number of reads")
    ax.set_ylabel("Observed richness")
    ax.set_title(title)
    return ax

def extract_sample_stats(metadata, sample):
    try:
        s_clean = metadata[metadata['relationships.run.data.id']==sample]['attributes.analysis-summary'].values[0].strip().rstrip(')"')
        lst = ast.literal_eval(s_clean)
    except AttributeError:
        lst = metadata[metadata['relationships.run.data.id']==sample]['attributes.analysis-summary'].values[0]
    df_tmp = pd.DataFrame(lst)
    total = int(df_tmp[df_tmp['key']=='Submitted nucleotide sequences']['value'].values[0])
    identified = (int(df_tmp[df_tmp['key']=='Predicted SSU sequences']['value'].values[0]) + 
                    int(df_tmp[df_tmp['key']=='Predicted LSU sequences']['value'].values[0]))
    ratio = (total - identified) / total
    return total, ratio


def plot_season_reads_hist(
    analysis_meta,
    samples_meta,
    name=None,
    use_robust_save=True,
    **kwargs,
) -> Dict[Dict, Dict, Dict, Dict]:
    total_dict = {'Spring': {}, 'Summer': {}, 'Autumn': {}, 'Winter': {}}

    # extracting the reads metadata per sample
    for sample in analysis_meta['relationships.run.data.id']:
        data_id = analysis_meta[analysis_meta['relationships.run.data.id']==sample]['relationships.sample.data.id'].values[0]
        season = samples_meta[samples_meta['id']==data_id]['season'].values[0]
        total_dict[season][sample] = extract_sample_stats(analysis_meta, sample)
    
    # plot histogram per season 
    fig, ax = plt.subplots(figsize=(10, 6))
    season_stats = {}
    for season, stats in total_dict.items():
        totals = [stats[0] for _, stats in total_dict[season].items()]  # stat contains (total, ratio)
        ax.hist(totals, bins=10, alpha=0.3, label=season)
        season_stats[season] = {
            'n_samples': len(totals),
            'mean_reads': np.mean(totals),
            'std_reads': np.std(totals),
            'min_reads': np.min(totals),
            'max_reads': np.max(totals)
        }
    
    ax.legend()
    ax.set_xlabel("Total reads per sample")
    ax.set_ylabel("Number of samples")
    ax.set_title("Distribution of Total Reads per Sample by Season")
    
    if use_robust_save:
        # Use new robust saving method
        save_plot_with_metadata(
            fig=fig,
            filename=name.replace('.png', '') if name else "season_reads_histogram",
            description=f"Histogram showing distribution of total sequencing reads per sample, grouped by collection season. Data from MGnify study {globals().get('analysisId', 'unknown')}. Each season shows different sequencing depth patterns.",
            plot_type="histogram_seasonal",
            data_info={
                "total_samples": len(analysis_meta),
                "seasons_analyzed": list(total_dict.keys()),
                "season_statistics": season_stats,
                "study_id": globals().get('analysisId', 'unknown')
            },
            **kwargs,
        )
    else:
        # Fallback to old method
        if name is not None:
            plt.savefig(os.path.join(OUT_FOLDER, name))
    
    plt.show()
    return total_dict

# -----------------------
# Plotting and saving
# -----------------------

def _json_serializer(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        return str(obj)

def _make_json_serializable(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: _make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_make_json_serializable(item) for item in obj)
    elif isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def save_plot_with_metadata(
    fig=None,
    filename=None,
    description="",
    plot_type="analysis",
    data_info=None,
    save_formats=None,
    out_dir=None,
    timestamp=True,
    dpi=300,
    bbox_inches='tight',
    **kwargs
):
    """
    Robustly save plots with comprehensive metadata and description.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure, optional
        Figure to save. If None, uses plt.gcf() (current figure).
    filename : str, optional
        Base filename (without extension). If None, auto-generates from plot_type and timestamp.
    description : str
        Textual description of the plot and what it shows.
    plot_type : str
        Type of plot (e.g., "rarefaction", "diversity", "taxonomy", "comparison").
    data_info : dict, optional
        Dictionary with information about the data used (shape, samples, etc.).
    save_formats : list, optional
        List of formats to save. Default: ['png', 'pdf'].
    out_dir : str or Path, optional
        Output directory. If None, uses global OUT_FOLDER or current directory.
    timestamp : bool
        Whether to include timestamp in filename. Default: True.
    dpi : int
        Resolution for raster formats. Default: 300.
    bbox_inches : str
        Bounding box setting. Default: 'tight'.
    **kwargs
        Additional arguments passed to fig.savefig().
    
    Returns
    -------
    dict
        Dictionary with saved file paths and metadata file path.
    
    Examples
    --------
    # Basic usage
    plt.plot([1, 2, 3], [1, 4, 9])
    result = save_plot_with_metadata(
        filename="quadratic_example",
        description="Simple quadratic function showing x² relationship",
        plot_type="example"
    )
    
    # With data info
    fig, ax = plt.subplots()
    ax.hist(data, bins=20)
    result = save_plot_with_metadata(
        fig=fig,
        filename="data_histogram",
        description="Distribution of sample values after preprocessing",
        plot_type="histogram",
        data_info={
            "n_samples": len(data),
            "mean": np.mean(data),
            "std": np.std(data),
            "range": [np.min(data), np.max(data)]
        }
    )
    """
    
    # Handle defaults
    if fig is None:
        fig = plt.gcf()
    
    if save_formats is None:
        save_formats = ['png', 'pdf']
    
    if out_dir is None:
        out_dir = globals().get('OUT_FOLDER', '.')
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        base_name = f"{plot_type}_plot"
    else:
        base_name = filename
    
    # Add timestamp if requested
    if timestamp:
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{base_name}_{time_str}"
    
    # Prepare metadata
    metadata = {
        "plot_info": {
            "filename_base": base_name,
            "description": description,
            "plot_type": plot_type,
            "created_at": datetime.now().isoformat(),
            "figure_size": fig.get_size_inches().tolist(),
            "dpi": int(dpi)
        },
        "data_info": _make_json_serializable(data_info or {}),
        "save_settings": {
            "formats": save_formats,
            "bbox_inches": bbox_inches,
            "dpi": int(dpi),
            "additional_kwargs": _make_json_serializable(kwargs)
        },
        "notebook_info": {
            "analysis_id": globals().get('analysisId', 'unknown'),
            "out_folder": str(out_dir),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        }
    }
    
    # Save plot in requested formats
    saved_files = {}
    for fmt in save_formats:
        plot_path = out_dir / f"{base_name}.{fmt}"
        fig.savefig(
            plot_path,
            format=fmt,
            dpi=dpi,
            bbox_inches=bbox_inches,
            **kwargs
        )
        saved_files[fmt] = str(plot_path)
        print(f"Saved {fmt.upper()}: {plot_path}")
    
    # Save metadata as JSON
    metadata_path = out_dir / f"{base_name}_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=_json_serializer)
    
    # Save description as text file
    desc_path = out_dir / f"{base_name}_description.txt"
    with open(desc_path, 'w', encoding='utf-8') as f:
        f.write(f"Plot Description\n")
        f.write(f"================\n\n")
        f.write(f"Filename: {base_name}\n")
        f.write(f"Type: {plot_type}\n")
        f.write(f"Created: {metadata['plot_info']['created_at']}\n\n")
        f.write(f"Description:\n{description}\n\n")
        if data_info:
            f.write(f"Data Information:\n")
            for key, value in data_info.items():
                f.write(f"  {key}: {value}\n")
    
    result = {
        "saved_files": saved_files,
        "metadata_file": str(metadata_path),
        "description_file": str(desc_path),
        "base_name": base_name
    }
    
    print(f"Metadata saved: {metadata_path}")
    print(f"Description saved: {desc_path}")
    
    return result


def save_current_plot(description, plot_type="analysis", **kwargs):
    """
    Convenience function to save the current matplotlib plot with description.
    
    Parameters
    ----------
    description : str
        Description of what the plot shows.
    plot_type : str
        Type of analysis or plot.
    **kwargs
        Additional arguments passed to save_plot_with_metadata.
    
    Returns
    -------
    dict
        Result from save_plot_with_metadata.
    """
    return save_plot_with_metadata(
        description=description,
        plot_type=plot_type,
        **kwargs
    )

def test_json_serialization():
    """Test function to verify JSON serialization works with numpy types."""
    test_data = {
        'numpy_int': np.int64(42),
        'numpy_float': np.float64(3.14),
        'numpy_array': np.array([1, 2, 3]),
        'nested_dict': {
            'numpy_val': np.int32(100),
            'regular_val': 'test'
        },
        'list_with_numpy': [np.float32(1.5), 'string', np.int16(5)]
    }
    
    serialized = _make_json_serializable(test_data)
    
    # Test that it can be serialized to JSON
    try:
        json_str = json.dumps(serialized, indent=2)
        print("JSON serialization test passed!")
        return True
    except TypeError as e:
        print(f"JSON serialization test failed: {e}")
        return False


# Updated function to integrate robust saving with existing rarefaction plots
def save_rarefaction_plot_with_metadata(fig, tax_levels, sample_type, table_shapes, description_suffix=""):
    """
    Save rarefaction plots with comprehensive metadata for taxonomic analysis.
    """
    description = f"""
    Rarefaction curves showing observed species richness versus sequencing depth across taxonomic levels.
    Analysis performed for {sample_type} samples at taxonomic levels: {', '.join(tax_levels)}.
    
    Each curve represents a different sample, with 10% prevalence filtering applied before rarefaction.
    Curves that plateau indicate sufficient sequencing depth for reliable diversity estimates.
    
    {description_suffix}
    """.strip()
    
    return save_plot_with_metadata(
        fig=fig,
        filename=f"rarefaction_{sample_type}_multilevel",
        description=description,
        plot_type="rarefaction_multilevel",
        data_info={
            "sample_type": sample_type,
            "taxonomic_levels": tax_levels,
            "table_shapes": table_shapes,
            "prevalence_cutoff": "10%",
            "analysis_type": "MGnify_taxonomic_profiling",
            "study_id": globals().get('analysisId', 'unknown')
        },
        save_formats=['png', 'pdf']
    )

# Function to save violin plots with metadata
def save_violin_plot_with_metadata(fig, plot_data, title, description):
    """Save violin/strip plots with metadata about taxonomic prevalence."""
    return save_plot_with_metadata(
        fig=fig,
        filename=f"taxonomic_prevalence_{plot_data.get('analysis_type', 'unknown')}",
        description=description,
        plot_type="violin_taxonomic_prevalence", 
        data_info=plot_data,
        save_formats=['png', 'pdf', 'svg']
    )