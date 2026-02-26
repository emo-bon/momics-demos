"""
Author: David Palecek
Date: 2026-02-26

This script generates combined metadata table for all the EMO-BON samples
which is supposed to replace the
"https://raw.githubusercontent.com/emo-bon/momics-demos/refs/heads/main/wf0_landing_page/emobon_sequencing_master.csv"

1. Clone all published station's RO-Crate repositories
2. Parse logsheets/transformed csv files to extract the sample sampling and measuring metadata.
3. Concatenate all the metadata into a single file and save it as a csv in this repository.

Use:

python combine_mgf_metadata.py
"""

import os
from pathlib import Path
import pandas as pd
import logging
import pickle
import requests

import json
import subprocess
import shutil
from utils import (
    retrieve_valid_stations,
    clone_repository,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TMP_DIR = Path("./tmp")

def get_transformed_csvs(path: Path) -> list[Path]:
    final_path = path / "logsheets" / "transformed" 
    if not os.path.isdir(final_path):
        logger.warning(f"No 'transformed' directory found in {path}. Skipping.")
        return []
    
    data = {}
    for file in (final_path).glob("*.csv"):
        try:
            df = pd.read_csv(file)
            data[file.stem] = df
            logger.info(f"Loaded {file} with shape {df.shape}")
        except Exception as e:
            logger.error(f"Failed to load {file}: {e}")
    return data

def process_environment_metadata(metadata: dict, env_type: str) -> pd.DataFrame:
    """
    Process metadata for a specific environment type (sediment or water).
    
    Args:
        metadata: Dictionary containing metadata dataframes
        env_type: Environment type ('sediment' or 'water')
        
    Returns:
        pd.DataFrame: Processed metadata with observatory info
    """
    observatory_key = f'{env_type}_observatory'
    measured_key = f'{env_type}_measured'
    sampling_key = f'{env_type}_sampling'
    
    if observatory_key not in metadata:
        logger.warning(f"No '{observatory_key}' metadata found. Skipping {env_type} processing.")
        return pd.DataFrame()
    
    df_measured = metadata[measured_key]
    df_sampling = metadata[sampling_key]

    # Merge with suffixes to handle overlapping columns
    df = pd.merge(df_measured, df_sampling, on='source_mat_id', how='inner', suffixes=('_x', '_y'))
    logger.info(f"Merged {env_type} measured and sampling metadata with shape {df.shape}")
    
    # Combine overlapping columns by coalescing (prefer non-NaN values)
    overlapping_cols = []
    for col in df.columns:
        if col.endswith('_x'):
            base_col = col[:-2]
            if f'{base_col}_y' in df.columns:
                overlapping_cols.append(base_col)
                # Coalesce: take non-NaN value from _x, otherwise from _y
                df[base_col] = df[col].combine_first(df[f'{base_col}_y'])
                df = df.drop(columns=[col, f'{base_col}_y'])
    
    if overlapping_cols:
        logger.info(f"Combined overlapping columns: {overlapping_cols}")

    if overlapping_cols != [] and overlapping_cols != ['source_mat_id_orig']:
        raise ValueError(f"columns not combined correctly: {overlapping_cols}")

    # add the observatory metadata
    df_obs = metadata[observatory_key]
    df[df_obs.columns] = df_obs.iloc[0].values  # add the observatory metadata to all rows

    return df
    

if __name__ == "__main__":
    owner = "emo-bon"
    url = f"https://api.github.com/orgs/{owner}/repos"

    if os.path.exists("scripts/valid_stations.pkl"):
        with open("scripts/valid_stations.pkl", "rb") as f:
            valid_stations = pickle.load(f)
        logger.info(f"Loaded {len(valid_stations)} valid stations from cache: {list(valid_stations.keys())}")
    else:
        valid_stations = retrieve_valid_stations(owner)
        with open("scripts/valid_stations.pkl", "wb") as f:
            pickle.dump(valid_stations, f)
        logger.info(f"Found {len(valid_stations)} valid stations: {list(valid_stations.keys())}")

    cloned_repos = {}
    for station_id, repo_url in valid_stations.items():
        logger.info(f"Cloning repository for station {station_id} from {repo_url}")
        repo = repo_url.split("/")[-1].replace(".git", "")
        repo_path = clone_repository(owner, repo=repo, target_dir=TMP_DIR)

        if repo_path:
            logger.info(f"Successfully cloned {repo} to {repo_path}")
            cloned_repos[station_id] = repo_path
    
    # --- Step 2: load all metadata per station ----
    logger.info("=" * 60)
    logger.info("Parsing metadata from cloned repositories...")
    logger.info("=" * 60)

    metadata = {}
    for station_id, repo_path in cloned_repos.items():
        logger.info(f"Parsing metadata for station {station_id} from {repo_path}")
        metadata[station_id] = get_transformed_csvs(repo_path)

        if 'sediment_observatory' in metadata[station_id]:
            logger.info(f"Found sediment_observatory metadata for station {station_id}")
            metadata[f"{station_id}-sed"] = process_environment_metadata(metadata[station_id], 'sediment')

        if 'water_observatory' in metadata[station_id]:
            logger.info(f"Found water_observatory metadata for station {station_id}")
            metadata[f"{station_id}-water"] = process_environment_metadata(metadata[station_id], 'water')

    # --- Step 3: concatenate all metadata into a single file -----
    logger.info("=" * 60)
    logger.info("Combining metadata into a single DataFrame...")
    logger.info("=" * 60)

    combined_df = pd.DataFrame()
    for key, df in metadata.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            combined_df = pd.concat([combined_df, df], ignore_index=True)
            logger.info(f"Added metadata from {key} with shape {df.shape}. Combined shape is now {combined_df.shape}")
        else:
            logger.warning(f"No valid DataFrame found for {key}. Skipping.")


    # data cleanup
    combined_df.drop_duplicates(subset='source_mat_id', inplace=True)
    logger.info(f"Removed duplicates based on 'source_mat_id'. Final shape is {combined_df.shape}")

    combined_df['depth'] = combined_df['depth'].astype(float)

    # save the combined metadata to a csv file
    output_path = Path("data/emo-bon_data/combined_metadata.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    logger.info(f"Saved combined metadata to {output_path} with shape {combined_df.shape}")





