# handlers.py

from typing import Callable, Optional
import pandas as pd
import logging

from .detectors import (
    is_emobon_processed,
    is_emobon_raw,
    is_mgnify_raw,
)
from .validators import (
    validate_taxonomy_processed,
    validate_abundance_ncbi,
    validate_abundance_no_ncbi,
)
from .converters import (
    emobon_raw_to_processed,
    mgnify_raw_to_processed,
)

logger = logging.getLogger(__name__)


class SourceHandler:
    """
    Defines the detection, validation, and optional conversion logic
    for a specific data source format.
    """
    def __init__(
        self,
        name: str,
        detect: Callable[[pd.DataFrame], bool],
        convert: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        validate: Optional[Callable[[pd.DataFrame], None]] = None,
    ):
        self.name = name
        self.detect = detect
        self.convert = convert
        self.validate = validate


# 🔹 Central registry of supported formats
SOURCE_HANDLERS = [
    SourceHandler(
        name="emobon_processed",
        detect=is_emobon_processed,
        validate=validate_taxonomy_processed,
    ),
    SourceHandler(
        name="emobon_raw",
        detect=is_emobon_raw,
        convert=emobon_raw_to_processed,
        validate=validate_taxonomy_processed,
    ),
    SourceHandler(
        name="mgnify_raw",
        detect=is_mgnify_raw,
        convert=mgnify_raw_to_processed,
    ),
]
