# taxonomy_table.py

from __future__ import annotations

import pandas as pd
import logging
from typing import Optional
from pydantic import BaseModel

from .sources.handlers import SOURCE_HANDLERS

logger = logging.getLogger(__name__)


# -------------------------
# Pydantic config validator
# -------------------------
class TaxonomyConfig(BaseModel):
    taxonomy_col: Optional[str] = None
    sample_col: Optional[str] = None
    taxon_col: Optional[str] = None
    source: Optional[str] = "unknown"


# -------------------------
# Main TaxonomyTable class
# -------------------------
class TaxonomyTable:
    """
    Main entry class for handling taxonomy tables and performing
    conversions, inference of source type, validations and pivoting.
    """

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

        self.source = self._infer_or_validate_source(cfg.source)

    # -------------------------
    # Source detection, validation and conversion
    # -------------------------
    def _infer_or_validate_source(self, source: str) -> str:
        # Case 1 — explicit user-provided source
        if source and source != "unknown":
            handler = next((h for h in SOURCE_HANDLERS if h.name == source), None)
            if handler is None:
                raise ValueError(f"Unknown source '{source}'")

            if handler.validate:
                handler.validate(self.df)

            return handler.name

        # Case 2 — auto-detect using handlers
        for handler in SOURCE_HANDLERS:
            if handler.detect(self.df):
                logger.info(f"Detected source: {handler.name}")

                if handler.convert:
                    self.df = handler.convert(self.df)

                return handler.name

        raise ValueError("Could not infer source type for the provided DataFrame.")

    # -------------------------
    # Example conversion method
    # -------------------------
    def to_abundance_table(self) -> pd.DataFrame:
        """
        Convert taxonomy table to some standard abundance table.
        Replace pivot_taxonomic_data_new with your real implementation.
        """
        from .abundance_table import AbundanceTable
        from .utils import pivot_taxonomic_data_new

        pivot = pivot_taxonomic_data_new(self.df)
        return AbundanceTable(pivot, source=self.source)
