import pandas as pd


def mgnify_raw_to_processed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw MGNify dataframe to standardized format.
    """
    # Rename columns
    df = df.rename(columns={"#SampleID": 'taxonomic_concat'})
    df.columns.name = "source material ID"
    df.set_index('taxonomic_concat', inplace=True)
    return df


def emobon_raw_to_processed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Insert your real transformation from raw → processed.
    """
    raise NotImplementedError("Conversion from emobon_raw is not implemented yet.")
