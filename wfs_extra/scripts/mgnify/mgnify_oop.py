from __future__ import annotations

from collections import defaultdict
import os
import json
import ast
import re
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
from typing import Iterable, Optional, Sequence, Tuple, Dict, Any, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AbundanceTable:
    """
    Handle taxonomy tables and common conversions (pivoting, aggregation, normalization).

    Expected common input layouts (examples):
      - long form: columns = ["sample", "taxon", "count", "taxonomy"] where `taxonomy` is
        a delimited string like "Kingdom;Phylum;Class;Order;Family;Genus;Species"
      - wide form: rows = taxa, columns = samples, values = counts (or vice-versa)

    Parameters
    ----------
    df : pd.DataFrame
        The underlying table of counts or taxonomy.
    taxonomy_col : Optional[str]
        Name of column that contains taxonomy strings (e.g. "taxonomy"); used for splitting.
    sample_col : Optional[str]
        Name of the sample identifier column in long form (e.g. "sample").
    taxon_col : Optional[str]
        Name of the taxon identifier column in long form (e.g. "taxon").
    """

    DEFAULT_RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

    def __init__(
            self, df: pd.DataFrame,
            source: str = "unknown",
            taxonomy_ranks: Optional[Iterable[str]] = None,
            taxonomy_col: str = None,
            index_col: Optional[str] = None,
            ncbi_col: str = "ncbi_tax_id",
            ):
        self.df = df
        self.taxonomy_ranks = taxonomy_ranks
        self.taxonomy_col = taxonomy_col
        self.index_col = index_col
        self.ncbi_col = ncbi_col
        self.source = self._check_source(source)

    # ---------------------
    # IO helpers
    # ---------------------
    @classmethod
    def from_csv(
        cls,
        path: Union[str, Path],
        sep: str = ",",
        source: str = "unknown",
        taxonomy_col: Optional[str] = None,
        **pd_read_csv_kwargs,
    ) -> AbundanceTable:
        """Factory classmethod to load table from CSV/TSV and return AbundanceTable instance."""
        path = Path(path)
        df = pd.read_csv(path, sep=sep, **pd_read_csv_kwargs)
        return cls(df, source, taxonomy_col=taxonomy_col)

    # ---------------------
    # Validation / conversion helpers
    # ---------------------
    def _check_source(self, source: str) -> None:
        """Basic checks on the source data based on known source types."""
        if source == "mgnify":
            if AbundanceTable.is_mgnify_raw(self.df):
                logger.info("raw MGNify data -> standardizing.")

            required_cols = {self.taxonomy_col, self.index_col, self.ncbi_col}
            missing = required_cols - set(self.df.columns)
            if missing:
                raise ValueError(f"Missing required columns for source 'mgnify': {missing}")
            
        elif source == "emobon":
            required_cols = {self.taxonomy_col, self.index_col}
            missing = required_cols - set(self.df.columns)
            if missing:
                raise ValueError(f"Missing required columns for source 'emobon': {missing}")

        return source
    
    @staticmethod
    def convert_mgnify_raw(df: pd.DataFrame, taxonomy_col: str = "taxonomy", index_col: str = "#SampleID", ncbi_col: str = "ncbi_tax_id") -> pd.DataFrame:
        """Convert raw MGNify dataframe to standardized format."""
        # Rename columns
        df = df.rename(columns={index_col: index_col, ncbi_col: ncbi_col})
        # Combine taxonomy columns into a single taxonomy string if needed
        if taxonomy_col not in df.columns:
            tax_cols = [col for col in df.columns if col.startswith("tax_")]
            df[taxonomy_col] = df[tax_cols].astype(str).agg(";".join, axis=1)
            df = df.drop(columns=tax_cols)
        return df

    @staticmethod
    def is_mgnify_raw(df: pd.DataFrame) -> bool:
        required_cols = {"#SampleID"}
        missing = required_cols - set(df.columns)
        starts_with_sk = bool(df['#SampleID'].str.startswith("sk__").all())
        unique_sample_ids = df['#SampleID'].is_unique
        return len(missing) == 0 and isinstance(df.index, pd.RangeIndex) and unique_sample_ids and starts_with_sk


    # @staticmethod
    # def is_emobon_raw(df: pd.DataFrame) -> bool:
    #     required_cols = {"sample_id", "taxonomy"}
    #     missing = required_cols - set(df.columns)
    #     starts_with_emobon = bool(df['sample_id'].str.startswith("emobon_").all())
    #     unique_sample_ids = df['sample_id'].is_unique
    #     return len(missing) == 0 and isinstance(df.index, pd.RangeIndex) and unique_sample_ids and starts_with_emobon


# class TaxonomyTable:
#     """
#     Handle taxonomy tables and common conversions (pivoting, aggregation, normalization).

#     Expected common input layouts (examples):
#       - long form: columns = ["sample", "taxon", "count", "taxonomy"] where `taxonomy` is
#         a delimited string like "Kingdom;Phylum;Class;Order;Family;Genus;Species"
#       - wide form: rows = taxa, columns = samples, values = counts (or vice-versa)

#     Parameters
#     ----------
#     df : pd.DataFrame
#         The underlying table of counts or taxonomy.
#     taxonomy_col : Optional[str]
#         Name of column that contains taxonomy strings (e.g. "taxonomy"); used for splitting.
#     sample_col : Optional[str]
#         Name of the sample identifier column in long form (e.g. "sample").
#     taxon_col : Optional[str]
#         Name of the taxon identifier column in long form (e.g. "taxon").
#     """

#     DEFAULT_RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

#     def __init__(
#         self,
#         df: pd.DataFrame,
#         taxonomy_col: Optional[str] = None,
#         sample_col: Optional[str] = None,
#         taxon_col: Optional[str] = None,
#     ):
#         self.df = df.copy()
#         self.taxonomy_col = taxonomy_col
#         self.sample_col = sample_col
#         self.taxon_col = taxon_col

#     # ---------------------
#     # IO helpers
#     # ---------------------
#     @classmethod
#     def from_csv(
#         cls,
#         path: Union[str, Path],
#         sep: str = ",",
#         taxonomy_col: Optional[str] = None,
#         sample_col: Optional[str] = None,
#         taxon_col: Optional[str] = None,
#         **pd_read_csv_kwargs,
#     ) -> "TaxonomyTable":
#         """Load table from CSV/TSV and return TaxonomyTable instance."""
#         path = Path(path)
#         df = pd.read_csv(path, sep=sep, **pd_read_csv_kwargs)
#         return cls(df, taxonomy_col=taxonomy_col, sample_col=sample_col, taxon_col=taxon_col)

#     def to_csv(self, path: Union[str, Path], sep: str = ",", index: bool = False, **kwargs) -> None:
#         """Save the underlying dataframe."""
#         path = Path(path)
#         path.parent.mkdir(parents=True, exist_ok=True)
#         self.df.to_csv(path, sep=sep, index=index, **kwargs)

#     # ---------------------
#     # Validation / conversion helpers
#     # ---------------------
#     def is_wide(self) -> bool:
#         """
#         Heuristic: wide if index or columns look like samples/taxa instead of 'long' triplet.
#         This is a simple guess; user should set sample_col/taxon_col for long form.
#         """
#         # If explicit sample_col/taxon_col provided, treat as long
#         if self.sample_col and self.taxon_col:
#             return False
#         # If more than 2 columns and a column named like 'sample' or 'taxon' -> long
#         cols_lower = [c.lower() for c in self.df.columns]
#         if any(name in cols_lower for name in ("sample", "taxon", "count", "abundance")):
#             return False
#         # Otherwise likely wide (samples as columns)
#         return True

#     # ---------------------
#     # Taxonomy splitting and normalization
#     # ---------------------
#     def split_taxonomy(
#         self,
#         taxonomy_col: Optional[str] = None,
#         sep: str = ";",
#         ranks: Optional[List[str]] = None,
#         fill: Optional[str] = None,
#         inplace: bool = True,
#     ) -> pd.DataFrame:
#         """
#         Split taxonomy string column into individual rank columns.

#         Returns the dataframe with added rank columns (or modifies self.df if inplace=True).

#         Parameters
#         ----------
#         taxonomy_col : str
#             name of column with taxonomy strings
#         sep : str
#             separator used in taxonomy strings (default ';')
#         ranks : list[str]
#             list of rank names to use (default DEFAULT_RANKS)
#         fill : str or None
#             value to fill missing ranks (None leaves NaN)
#         """
#         taxonomy_col = taxonomy_col or self.taxonomy_col
#         if taxonomy_col is None:
#             raise ValueError("taxonomy_col must be provided (either at init or here).")
#         ranks = ranks or self.DEFAULT_RANKS

#         # Split into components
#         split_df = self.df[taxonomy_col].astype(str).str.split(sep, expand=True)
#         # Trim whitespace
#         split_df = split_df.applymap(lambda x: x.strip() if pd.notna(x) else x)

#         # Keep only as many ranks as provided (or extend with None)
#         # Rename columns to desired rank names
#         for i, rank in enumerate(ranks):
#             if i in split_df.columns:
#                 self.df[rank] = split_df[i]
#             else:
#                 self.df[rank] = fill

#         return self.df if inplace else self.df.copy()

#     def collapse_taxonomy_to_rank(
#         self,
#         rank: str,
#         taxonomy_ranks: Optional[List[str]] = None,
#         new_taxon_col: str = "taxon_at_rank",
#         fill_with: str = "unassigned",
#     ) -> pd.DataFrame:
#         """
#         Create/replace a taxon column collapsed at the specified rank.
#         For example, collapse at 'genus' will set taxon_at_rank to the genus name.
#         """
#         taxonomy_ranks = taxonomy_ranks or self.DEFAULT_RANKS
#         if rank not in taxonomy_ranks:
#             raise ValueError(f"rank '{rank}' not in taxonomy_ranks: {taxonomy_ranks}")

#         # Ensure rank columns exist
#         missing = [r for r in taxonomy_ranks if r not in self.df.columns]
#         if missing:
#             # if taxonomy_col exists, try to split it
#             if self.taxonomy_col:
#                 self.split_taxonomy(taxonomy_col=self.taxonomy_col, ranks=taxonomy_ranks)
#             else:
#                 # create empty columns to avoid KeyError
#                 for r in missing:
#                     self.df[r] = None

#         # collapse: choose value at rank or fallback up the hierarchy
#         def choose_taxon(row):
#             val = row.get(rank)
#             if pd.notna(val) and str(val).strip():
#                 return val
#             # fallback: search from higher ranks downwards
#             idx = taxonomy_ranks.index(rank)
#             for j in range(idx - 1, -1, -1):
#                 v = row.get(taxonomy_ranks[j])
#                 if pd.notna(v) and str(v).strip():
#                     return v
#             return fill_with

#         self.df[new_taxon_col] = self.df.apply(choose_taxon, axis=1)
#         return self.df

#     # ---------------------
#     # Pivoting / reshaping
#     # ---------------------
#     def long_to_wide(
#         self,
#         sample_col: Optional[str] = None,
#         taxon_col: Optional[str] = None,
#         value_col: str = "count",
#         aggfunc=sum,
#         fill_value: Optional[float] = 0.0,
#     ) -> pd.DataFrame:
#         """
#         Convert a long-form table (rows = sample x taxon x value) into a wide table
#         with rows as taxa and columns as samples (or vice-versa depending on preference).
#         Returns the wide DataFrame (does not modify self.df).
#         """
#         sample_col = sample_col or self.sample_col
#         taxon_col = taxon_col or self.taxon_col
#         if sample_col is None or taxon_col is None:
#             raise ValueError("sample_col and taxon_col must be provided for long_to_wide.")

#         pivot = (
#             self.df
#             .pivot_table(index=taxon_col, columns=sample_col, values=value_col, aggfunc=aggfunc, fill_value=fill_value)
#             .sort_index()
#         )
#         # Optionally flatten columns (if MultiIndex)
#         if isinstance(pivot.columns, pd.MultiIndex):
#             pivot.columns = [str(c) for c in pivot.columns]
#         return pivot

#     def wide_to_long(
#         self,
#         index_is_taxa: bool = True,
#         sample_name: str = "sample",
#         taxon_name: str = "taxon",
#         value_name: str = "count",
#         dropna: bool = True,
#     ) -> pd.DataFrame:
#         """
#         Convert a wide table (samples in columns, taxa as index or samples as index) into long form.
#         Returns a long DataFrame with columns [sample_name, taxon_name, value_name].
#         """
#         df = self.df.copy()
#         if index_is_taxa:
#             df = df.reset_index()
#             long = df.melt(id_vars=df.columns[0], var_name=sample_name, value_name=value_name)
#             long = long.rename(columns={df.columns[0]: taxon_name})
#         else:
#             # taxa are columns, samples are index
#             long = df.reset_index().melt(id_vars=df.index.name or "index", var_name=taxon_name, value_name=value_name)
#             long = long.rename(columns={long.columns[0]: sample_name})
#         if dropna:
#             long = long.dropna(subset=[value_name])
#         return long

#     # ---------------------
#     # Aggregation, filtering, normalization
#     # ---------------------
#     def aggregate_at_rank(
#         self,
#         wide_table: pd.DataFrame,
#         rank_map: Optional[Dict[str, str]] = None,
#         new_index_name: str = "taxon_at_rank",
#         aggfunc=sum,
#     ) -> pd.DataFrame:
#         """
#         Aggregate a wide table (index = taxon identifiers) to a coarser rank.

#         Parameters
#         ----------
#         wide_table : pd.DataFrame
#             wide table index should be taxon identifiers that appear in self.df (or in taxon_col)
#         rank_map : dict
#             mapping from original taxon_id -> rank name (e.g. genus). If not provided, the function
#             attempts to use self.df to build the mapping based on taxon_col and a rank column (e.g. 'genus').
#         """
#         if rank_map is None:
#             if self.taxon_col is None:
#                 raise ValueError("Either rank_map must be provided or taxon_col must be set on the instance.")
#             # Build mapping from taxon identifier to the desired rank column
#             if new_index_name not in self.df.columns:
#                 raise ValueError(f"new_index_name '{new_index_name}' not found in self.df; run collapse_taxonomy_to_rank first.")
#             mapping = self.df.set_index(self.taxon_col)[new_index_name].to_dict()
#         else:
#             mapping = rank_map

#         # Ensure every index value has a mapping; map missing to 'unassigned'
#         mapped = [mapping.get(idx, "unassigned") for idx in wide_table.index]
#         wide_table = wide_table.copy()
#         wide_table[new_index_name] = mapped
#         aggregated = wide_table.groupby(new_index_name).aggregate(aggfunc)
#         return aggregated

#     def to_relative_abundance(self, wide_table: pd.DataFrame, axis: int = 0) -> pd.DataFrame:
#         """
#         Convert absolute counts to relative abundance.
#         axis=0 -> columns are samples (normalize columns to sum 1)
#         axis=1 -> rows are samples
#         """
#         df = wide_table.copy().astype(float)
#         if axis == 0:
#             col_sums = df.sum(axis=0)
#             # avoid division by zero
#             col_sums[col_sums == 0] = 1.0
#             return df.div(col_sums, axis=1)
#         elif axis == 1:
#             row_sums = df.sum(axis=1)
#             row_sums[row_sums == 0] = 1.0
#             return df.div(row_sums, axis=0)
#         else:
#             raise ValueError("axis must be 0 or 1")

#     def filter_low_abundance(
#         self,
#         wide_table: pd.DataFrame,
#         min_count: Optional[float] = None,
#         min_relative: Optional[float] = None,
#         axis: int = 0,
#     ) -> pd.DataFrame:
#         """
#         Remove taxa (or samples) below thresholds.
#         axis=0 -> filter taxa by their max or mean across samples
#         axis=1 -> filter samples by their max or mean across taxa
#         """
#         df = wide_table.copy()
#         if min_count is not None:
#             if axis == 0:
#                 keep = (df.max(axis=1) >= min_count)
#                 return df.loc[keep, :]
#             else:
#                 keep = (df.max(axis=0) >= min_count)
#                 return df.loc[:, keep]
#         if min_relative is not None:
#             rel = self.to_relative_abundance(df, axis=0)
#             keep = (rel.max(axis=1) >= min_relative)
#             return df.loc[keep, :]
#         return df

#     # ---------------------
#     # Utility / helpers
#     # ---------------------
#     def summarize(self) -> pd.DataFrame:
#         """Quick summary: number of rows, columns, sample/taxon info if possible."""
#         info = {
#             "n_rows": len(self.df),
#             "n_columns": len(self.df.columns),
#             "columns": list(self.df.columns),
#             "is_wide_guess": self.is_wide(),
#         }
#         return pd.DataFrame.from_dict(info, orient="index", columns=["value"])

#     # ---------------------
#     # Example usage helpers (not modifying self.df unless intended)
#     # ---------------------
#     @staticmethod
#     def example_long_df() -> pd.DataFrame:
#         """Return a tiny example long-form dataframe for tests / illustration."""
#         return pd.DataFrame(
#             {
#                 "sample": ["s1", "s1", "s2", "s2"],
#                 "taxon": ["t1", "t2", "t1", "t3"],
#                 "count": [10, 5, 2, 7],
#                 "taxonomy": [
#                     "Bacteria;Proteobacteria;Gammaproteobacteria",
#                     "Bacteria;Firmicutes;Bacilli",
#                     "Bacteria;Proteobacteria;Gammaproteobacteria",
#                     "Bacteria;Actinobacteria;Actinobacteria",
#                 ],
#             }
#         )

# # ---------------------
# # Usage example
# # ---------------------
# if __name__ == "__main__":
#     # Create from example
#     tt = TaxonomyTable(TaxonomyTable.example_long_df(), taxonomy_col="taxonomy", sample_col="sample", taxon_col="taxon")

#     # Split taxonomy strings into rank columns
#     tt.split_taxonomy(sep=";")

#     # Collapse to genus-like rank (here "class" in example)
#     tt.collapse_taxonomy_to_rank(rank="class", new_taxon_col="taxon_at_class")

#     # Convert long -> wide (taxa x samples)
#     wide = tt.long_to_wide(value_col="count")
#     print("Wide table:\n", wide)

#     # Aggregate to rank using the new_taxon_col mapping
#     # Build a rank_map from the long df
#     rank_map = tt.df.set_index("taxon")["taxon_at_class"].to_dict()
#     agg = tt.aggregate_at_rank(wide, rank_map=rank_map, new_index_name="taxon_at_class")
#     print("Aggregated at class:\n", agg)

#     # Relative abundance (columns are samples)
#     rel = tt.to_relative_abundance(agg, axis=0)
#     print("Relative abundance:\n", rel)
