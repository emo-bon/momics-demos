# base.py
from __future__ import annotations

import pandas as pd
import logging
from typing import Optional, Union
from pathlib import Path
from pydantic import BaseModel

from .sources.handlers import SOURCE_HANDLERS
from wfs_extra.scripts.mgnify.tables.sources.validators import validate_abundance_ncbi
# from .sources.converters import mgnify_raw_to_processed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]


# -------------------------
# Pydantic config validator
# -------------------------
class TaxonomyConfig(BaseModel):
    taxonomy_col: Optional[str] = None
    sample_col: Optional[str] = None
    taxon_col: Optional[str] = None
    source: Optional[str] = "unknown"


class AbundanceConfig(BaseModel):
    taxonomy_col: Optional[str] = None
    source: Optional[str] = "unknown"


class BaseTable:
    def __init__(self, df: pd.DataFrame, source: Optional[str] = "unknown"):
        self.df = df
        self.source = self._infer_or_validate_source(source)

    def _infer_or_validate_source(self, source: str) -> str:
        # explicit override
        if source != "unknown":
            handler = next((h for h in SOURCE_HANDLERS if h.name == source), None)
            if handler is None:
                raise ValueError(f"Unknown source '{source}'")
            if handler.validate:
                logger.info(f"Validating source: {handler.name}")
                handler.validate(self.df)
            return handler.name

        # Case 2 — auto-detect using handlers
        for handler in SOURCE_HANDLERS:
            logger.info(f"Trying to infer source: {handler.name}")
            if handler.detect(self.df):
                logger.info(f"Detected source: {handler.name}")
    
                if handler.convert:
                    self.df = handler.convert(self.df)
                return handler.name

        raise ValueError("Could not infer data source type.")
    
    # -------------------------
    # IO HELPERS
    # -------------------------
    @classmethod
    def from_csv(
        cls,
        path: Union[str, Path],
        sep: str = ",",
        source: str = "unknown",
        **kwargs,
    ):
        path = Path(path)
        df = pd.read_csv(path, sep=sep, **kwargs)
        return cls(df, source=source)


class TaxonomyTable(BaseTable):
    def __init__(
        self,
        df: pd.DataFrame,
        taxonomy_col: Optional[str] = None,
        sample_col: Optional[str] = None,
        taxon_col: Optional[str] = None,
        source: Optional[str] = "unknown",
    ):
        self.df = df
        self.taxonomy_col = taxonomy_col
        self.sample_col = sample_col
        self.taxon_col = taxon_col

        cfg = TaxonomyConfig(
            taxonomy_col=taxonomy_col,
            sample_col=sample_col,
            taxon_col=taxon_col,
            source=source,
        )

        super().__init__(df, cfg.source)

    # -------------------------
    # conversion to abundance table
    # -------------------------
    def to_abundance_table(self) -> pd.DataFrame:
        """
        Convert taxonomy table to some standard abundance table.
        Replace pivot_taxonomic_data_new with your real implementation.
        """
        from wfs_extra.scripts.mgnify.mgnify_utils import pivot_taxonomic_data_new

        pivot = pivot_taxonomic_data_new(self.df)
        return AbundanceTable(pivot, source=self.source)


class AbundanceTable(BaseTable):
    def __init__(self, df, source="unknown", taxonomy_ranks=None, taxonomy_col=None):
        self.taxonomy_ranks = taxonomy_ranks or DEFAULT_RANKS
        self.taxonomy_col = taxonomy_col

        cfg = AbundanceConfig(
            taxonomy_col=taxonomy_col,
            source=source,
        )

        super().__init__(df, cfg.source)


    def to_taxonomy_table(self, drop_zeros: bool = True) -> pd.DataFrame:
        """Invert processed mgnify abundance table, ie pivoted taxonomy
        back to long format compatible with TaxonomyTable."""
        # validate firs the mgnify processed format
        from .sources.validators import validate_abundance_ncbi, validate_abundance_no_ncbi

        if validate_abundance_ncbi(self.df):
            logger.info("DataFrame is abundance with ncbi_tax_id data.")
            flag_ncbi = True
        elif validate_abundance_no_ncbi(self.df):
            logger.warning("DataFrame is abundance without ncbi_tax_id data.")
            flag_ncbi = False
        else:
            raise ValueError("DataFrame is not in valid Abundance processed format.")

        reset = self.df.reset_index()

        if flag_ncbi:
            assert "ncbi_tax_id" in reset.columns, "Expected 'ncbi_tax_id' column in DataFrame."
            melted = reset.melt(
                id_vars=[c for c in ["taxonomic_concat", "ncbi_tax_id"] if c in reset.columns],
                value_vars=[c for c in reset.columns if c not in ["taxonomic_concat", "ncbi_tax_id"]],
                var_name='source material ID',
                value_name="abundance",
            )
            melted = melted.set_index(['source material ID', 'ncbi_tax_id'])
        else:
            melted = reset.melt(
                id_vars=[c for c in ["taxonomic_concat"] if c in reset.columns],
                value_vars=[c for c in reset.columns if c not in ["taxonomic_concat"]],
                var_name='source material ID',
                value_name="abundance",
            )
            melted = melted.set_index(['source material ID'])

        if drop_zeros:
            melted = melted[melted["abundance"] != 0].copy()

        try:
            melted["abundance"] = melted["abundance"].astype(int)
        except Exception:
            melted["abundance"] = pd.to_numeric(melted["abundance"], errors="coerce")
        logger.info(f"self.source before conversion to Taxonomy table {self.source}")
        return melted, flag_ncbi, TaxonomyTable(melted, source=self.source)
