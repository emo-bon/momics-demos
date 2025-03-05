#!/usr/bin/python

"""
This is refactored code from the `landing_page_sandbox.ipynb` notebook to streamline
the update of the `min_merged.csv` file which serves as a basis for the dashboard.

There is no GH action or any scheduler to run this regularly.

Notes related to improvements:

# min version
# columns from shipment: source_mat_id, scientific_name, ref_code
# this should run through the validation process

# columns from MGflow tracker:
#   ref_code (to merge on), batch_number,
#   seq_run_ro_crate_fname, forward_read_fname, backward_read_fname, run_status,
#   version, date_started, who, system_run, output_loc, output_size, notes
"""

import os
import sys
import pandas as pd
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils import setup_local, reconfig_logger

CSV_LOCAL_PATH = Path(__file__).parent.joinpath("min_merged.csv")
ALL_SHIPMENTS = ["001", "002", "003-0", "003-1", "003-2"]


def get_seq_track_data(kind: str = "FILTERS") -> pd.DataFrame:
    """
    Fetch sequencing tracking data from a Google Sheets document.

    Parameters:
    kind (str): The sheet name to fetch data from. Default is "FILTERS".

    Returns:
    pd.DataFrame: Processed sequencing tracking data.
    """
    url = f"https://docs.google.com/spreadsheets/d/1j9tRRsRCcyViDMTB1X7lx8POY1P5bV7UijxKKSebZAM/gviz/tq?tqx=out:csv&sheet={kind}"
    df = pd.read_csv(url)

    df_out = process_seq_track_data(df, kind)
    return df_out


def process_seq_track_data(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """
    Process sequencing tracking data by renaming and dropping columns.

    Parameters:
    df (pd.DataFrame): The raw data frame to process.
    kind (str): The type of data being processed.

    Returns:
    pd.DataFrame: Processed data frame.
    """
    df_out = df[df.columns[:17]]  # drop empty cols at the end
    df_out.rename(columns={"Unnamed: 16": "comments"}, inplace=True)
    try:
        df_out.drop(columns=["Comp. Resources", "Run Duration"], inplace=True)
    except KeyError:
        pass
    return df_out


def query_batch_shipment_data(batch_string: str) -> pd.DataFrame:
    """
    Fetch and process shipment data for a specific batch.

    Parameters:
    batch_string (str): The batch identifier.

    Returns:
    pd.DataFrame: Processed shipment data for the specified batch.
    """
    url = f"https://raw.githubusercontent.com/emo-bon/sequencing-crate/refs/heads/main/shipment/batch-{batch_string}/run-information-batch-{batch_string}.csv"
    df = pd.read_csv(url)
    df["batch"] = batch_string

    # Extract the sample type
    if "source_material_id" in df.columns:
        for i in range(len(df)):
            if "_Wa_" in df.loc[i, "source_material_id"]:
                df.loc[i, "sample_type"] = "filters"
            else:
                df.loc[i, "sample_type"] = "sediment"

            if "blank" in df.loc[i, "source_material_id"].lower():
                df.loc[i, "sample_type"] = df.loc[i, "sample_type"] + "_blank"
    else:
        for i in range(len(df)):
            if "_Wa_" in df.loc[i, "old_source_mat_id"]:
                df.loc[i, "sample_type"] = "filters"
            else:
                df.loc[i, "sample_type"] = "sediment"

            if "blank" in df.loc[i, "old_source_mat_id"].lower():
                df.loc[i, "sample_type"] = df.loc[i, "sample_type"] + "_blank"

    return df


def query_all_shipment_data() -> pd.DataFrame:
    """
    Fetch and process shipment data for all batches.

    Returns:
    pd.DataFrame: Combined shipment data for all batches.
    """
    df = pd.concat(
        [query_batch_shipment_data(batch) for batch in ALL_SHIPMENTS], ignore_index=True
    )
    return df


def query_track_data() -> pd.DataFrame:
    """
    Fetch and process sequencing tracking data for sediments and filters.

    Returns:
    pd.DataFrame: Combined sequencing tracking data.
    """
    df_sed = get_seq_track_data("SEDIMENTS")
    df_filt = get_seq_track_data("FILTERS")
    df_mocks = get_seq_track_data("MOCKS")
    # df_mocks[["Comp. Resources", "Run Duration"]] = None
    df_blanks = get_seq_track_data("BLANKS")

    # Concatenate the two dataframes
    df = pd.concat([df_sed, df_filt, df_mocks, df_blanks],
                   ignore_index=True)

    # Rename certain columns
    df.rename(columns={"Seq Run RO-Crate Filename": "seq_run_ro_crate_fname"},
              inplace=True,)
    df.rename(columns={"Forward Read Filename": "forward_read_fname"}, inplace=True)
    df.rename(columns={"BackwardRead Filename": "backward_read_fname"}, inplace=True)
    df.rename(columns={"Output Location": "output_loc"}, inplace=True)

    # Rename columns to replace the space with underscore and make them lowercase
    df.columns = df.columns.str.replace(" ", "_").str.lower()
    return df


def infer_sample_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer the sample type based on the source material ID.

    Parameters:
    df (pd.DataFrame): The data frame containing source material IDs.

    Returns:
    pd.DataFrame: Data frame with inferred sample types.
    """
    df["sample_type"] = df["old_source_mat_id"].apply(
        lambda x: "filters" if "_Wa_" in x else "sediment"
    )
    df["sample_type"] = df["sample_type"].apply(
        lambda x: x + "_blank" if "blank" in df["source_mat_id_orig"].str.lower() else x
    )
    return df


def min_merge(df_shipment: pd.DataFrame, df_tracking: pd.DataFrame) -> pd.DataFrame:
    """
    Merge shipment and tracking data on the reference code.

    Parameters:
    df_shipment (pd.DataFrame): The shipment data frame.
    df_tracking (pd.DataFrame): The tracking data frame.

    Returns:
    pd.DataFrame: Merged data frame.
    """
    df_shipment["ref_code"] = df_shipment["ref_code"].str.replace(" ", "")
    df_tracking["ref_code"] = df_tracking["ref_code"].str.replace(" ", "")
    try:
        df_shipment["obs_id"] = df_shipment["source_mat_id"].str.split("_").str[1]
    except KeyError:
        df_shipment["obs_id"] = df_shipment["source_material_id"].str.split("_").str[1]

    df_shipment = df_shipment[["ref_code", "obs_id", "batch", "sample_type", "reads_name"]]

    df_tracking = df_tracking[
        [
            "ref_code",
            "seq_run_ro_crate_fname",
            "forward_read_fname",
            "backward_read_fname",
            "run_status",
            "version",
            "date_started",
            "who",
            "system_run",
            "output_loc",
            "output_size",
        ]
    ]

    # here I want to add obs_id from the 

    df = df_shipment.merge(df_tracking, on="ref_code", how="left")
    df.set_index("ref_code", inplace=True)
    return df


# These funcs are duplicated of utils in the dev branch of the py-udal-mgo
# https://github.com/fair-ease/py-udal-mgo
def rewrite_file(data: pd.DataFrame, path: str) -> None:
    """
    Ask for confirmation and rewrite the file.

    Parameters:
    data (pd.DataFrame): The data frame to write to the file.
    path (str): The file path to write the data to.
    """
    if input("Overwrite the file with the current version? [Y/n] ").lower() == "y":
        data.to_csv(path)


def check_diffs(data: pd.DataFrame, path: str, logger) -> bool:
    """
    Check differences between the current and the previous version of the file.

    Parameters:
    data (pd.DataFrame): The current data frame to compare.
    path (str): The file path of the previous version of the data frame.
    logger: The logger object to log information.

    Returns:
    bool: True if there are differences between the current and the previous version of the file, False otherwise.
    """
    # if the file does not exist, there is no need to show the differences
    if not os.path.exists(path):
        logger.info("Observatories file does not exist, saving pulled one.")
        data.to_csv(path)
        return False

    previous = pd.read_csv(path, index_col=[0])

    logger.info("Comparing the current and the previous version of the file...")

    diffs = data.compare(previous, result_names=("current", "previous"))
    if diffs.empty:
        logger.info(
            "No differences between the current and the previous version of the file."
        )
        return False

    logger.info("Differences between the current and the previous version of the file:")
    logger.info(diffs)
    return True


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    reconfig_logger(level=logging.INFO)

    df_shipments = query_all_shipment_data()
    df_tracking = query_track_data()

    df = min_merge(df_shipments, df_tracking)
    if check_diffs(df, path=CSV_LOCAL_PATH, logger=logger):
        rewrite_file(df, CSV_LOCAL_PATH)
        print("min_merged.csv updated")
