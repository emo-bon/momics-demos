"""
Author: David Palecek
Date: 2026-02-25

This script generates concatenated parquet files of all the main metaGOflow analysis tables

1. Clone all published metaGOflow RO-Crate repositories
2. Parse ro-crate-metadata.json files to extract download URLs
3. Download tables: SSU, LSU, GO-slim, GO, Kegg, Pfam
4. Concatenate all the tables and save them as parquet files in the data directory for use in the notebooks

Use:

python combine_mgf_results.py
"""

from pathlib import Path
import pandas as pd
import logging
import requests
import json
import shutil
from utils import (
    clone_repository,
    retrieve_valid_gh_analysis_clusters,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#--------------
# Setup
#--------------
TMP_DIR = Path("./tmp")
OUTPUT_DIR = Path("./data/emo-bon_data")


def find_rocrate_metadata_files(repo_path: Path) -> list[Path]:
    """
    Find all ro-crate-metadata.json files in a repository.
    
    Args:
        repo_path: Path to the local repository
        
    Returns:
        list[Path]: List of paths to ro-crate-metadata.json files
    """
    metadata_files = list(repo_path.glob("**/ro-crate-metadata.json"))
    logger.info(f"Found {len(metadata_files)} ro-crate-metadata.json files in {repo_path.name}")
    return metadata_files


def extract_download_urls(metadata_file: Path, name_patterns: list[str]) -> dict[str, str]:
    """
    Parse ro-crate-metadata.json and extract download URLs for files matching name patterns.
    
    Args:
        metadata_file: Path to ro-crate-metadata.json
        name_patterns: List of name patterns to match (e.g., ["Tab-separated formatted taxon counts for LSU sequences"])
        
    Returns:
        dict: Mapping of file type (LSU, SSU, etc.) to download URL
    """
    try:
        with open(metadata_file, 'r') as f:
            data = json.load(f)
        
        download_urls = {}
        
        # Iterate through @graph to find files with matching names
        for item in data.get('@graph', []):
            name = item.get('name', '')
            download_url = item.get('downloadUrl', '')
            
            if name and download_url:
                # Check if name matches any of our patterns
                for pattern in name_patterns:
                    if pattern in name:
                        # Extract file type (LSU, SSU, etc.)
                        if 'LSU' in name:
                            file_type = 'LSU'
                        elif 'SSU' in name:
                            file_type = 'SSU'
                        elif 'InterProScan slim' in name:
                            file_type = 'go_slim'
                        elif 'GO summary' in name:
                            file_type = 'go'
                        elif 'InterProScan' in name:
                            file_type = 'ips'
                        elif "KO summary" in name:
                            file_type = 'ko'
                        elif 'PFAM summary' in name:
                            file_type = 'pfam'
                        else:
                            file_type = 'unknown'
                            logger.info('============')
                            logger.debug(f"{name} does not match known file types, assigning 'unknown'")
                            logger.info('============')
                        
                        download_urls[file_type] = download_url
                        logger.debug(f"Found {file_type}: {download_url}")
        
        return download_urls
        
    except Exception as e:
        logger.error(f"Failed to parse {metadata_file}: {e}")
        return {}


def download_file_from_url(url: str, output_path: Path) -> bool:
    """
    Download a file from a URL.
    
    Args:
        url: URL to download from
        output_path: Local path to save the file
        
    Returns:
        bool: True if download succeeded, False otherwise
    """
    try:
        logger.info(f"Downloading from {url}")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to file
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"Successfully downloaded to {output_path} ({len(response.content)} bytes)")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download from {url}: {e}")
        return False


def process_rocrate_folder(metadata_file: Path, output_dir: Path, name_patterns: list[str]) -> dict[str, Path]:
    """
    Process a single ro-crate folder: extract URLs and download files.
    
    Args:
        metadata_file: Path to ro-crate-metadata.json
        output_dir: Directory to save downloaded files
        name_patterns: List of file name patterns to download
        
    Returns:
        dict: Mapping of file types to their local paths
    """
    # Extract the ro-crate folder name (e.g., "EMOBON_ROSKOGO_So_2-ro-crate")
    rocrate_folder = metadata_file.parent.name
    
    # Extract download URLs
    download_urls = extract_download_urls(metadata_file, name_patterns)
    
    if not download_urls:
        logger.warning(f"No matching files found in {rocrate_folder}")
        return {}
    
    # Download each file
    downloaded_files = {}
    for file_type, url in download_urls.items():
        # Create output path: output_dir/file_type/rocrate_folder_file_type.tsv
        filename = f"{rocrate_folder}_{file_type}.tsv"
        output_path = output_dir / file_type / filename
        
        if download_file_from_url(url, output_path):
            downloaded_files[file_type] = output_path
    
    return downloaded_files


def process_taxonomy_columns(df: pd.DataFrame) -> None:
    pass

def load_tsv_to_dataframe(file_path: Path, file_type) -> pd.DataFrame | None:
    """
    Load a TSV file into a pandas DataFrame.
    
    Args:
        file_path: Path to the TSV file
        
    Returns:
        pd.DataFrame: Loaded DataFrame, or None if loading failed
    """
    logger.info(f"Loading {file_path} of file type {file_type}")
    if file_type in ['go', 'go_slim']:
        df = pd.read_csv(file_path, sep=',', names=['id', 'name', 'aspect', 'abundance'])
    elif file_type in ['ips']:
        df = pd.read_csv(file_path, sep=',', names=['abundance', 'accession', 'description'])
        df = df[['accession', 'description', 'abundance']]  # Move accession to front
    elif file_type in ['ko', 'pfam']:
        df = pd.read_csv(file_path, sep=',', names=['abundance', 'entry', 'name'])
        df = df[['entry', 'name', 'abundance']]  # Move entry to front
    else:
        df = pd.read_csv(file_path, sep='\t', skiprows=1)  # For LSU and SSU, skip the first row which is a comment
        process_taxonomy_columns(df)
    return df

def concatenate_files_by_type(output_dir: Path, file_type: str, save_format: str = 'parquet') -> pd.DataFrame | None:
    """
    Concatenate all TSV files of a specific type into a single DataFrame.
    
    Args:
        output_dir: Directory containing downloaded files organized by type
        file_type: Type of file to concatenate (e.g., 'LSU', 'SSU', 'GO', 'Kegg', 'Pfam')
        save_format: Format to save combined file ('parquet', 'tsv', or 'both')
        
    Returns:
        pd.DataFrame: Combined dataframe, or None if no files found
    """
    # Find all TSV files in subdirectories for this type
    tsv_files = []
    for cluster_dir in output_dir.glob("cluster_*"):
        cluster_type_dir = cluster_dir / file_type
        if cluster_type_dir.exists():
            tsv_files.extend(list(cluster_type_dir.glob("*.tsv")))
    
    if not tsv_files:
        logger.warning(f"No TSV files found for type: {file_type}")
        return None
    
    logger.info(f"Found {len(tsv_files)} {file_type} files to concatenate")
    
    # Read and concatenate all files
    dfs = []
    for tsv_file in tsv_files:
        try:
            logger.debug(f"Reading {tsv_file.name}")
            df = load_tsv_to_dataframe(tsv_file, file_type)
            df['source_mat_id'] = str(tsv_file.stem).split('-')[0]  # Add source column for traceability
            df = df[['source_mat_id'] + [col for col in df.columns if col != 'source_mat_id']]  # Move source to front
            
            dfs.append(df)
            
        except Exception as e:
            logger.error(f"Failed to read {tsv_file}: {e}")
            continue
    
    if not dfs:
        logger.error(f"Failed to read any {file_type} files")
        return None
    
    # Concatenate all dataframes
    logger.info(f"Concatenating {len(dfs)} dataframes for {file_type}...")
    combined_df = pd.concat(dfs, ignore_index=True)
    
    logger.info(f"Combined {file_type} dataframe: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns")
    
    # Save the combined file
    if save_format in ['parquet', 'both']:
        parquet_path = output_dir / f"metagoflow_analyses.{file_type}.parquet"
        combined_df.to_parquet(parquet_path, index=False)
        logger.info(f"Saved combined {file_type} to {parquet_path}")
    
    if save_format in ['tsv', 'both']:
        tsv_path = output_dir / f"metagoflow_analyses.{file_type}.tsv"
        combined_df.to_csv(tsv_path, sep='\t', index=False)
        logger.info(f"Saved combined {file_type} to {tsv_path}")
    
    return combined_df


def concatenate_all_types(output_dir: Path, file_types: list[str] = None, save_format: str = 'parquet') -> dict[str, pd.DataFrame]:
    """
    Concatenate all file types found in the output directory.
    
    Args:
        output_dir: Directory containing downloaded files
        file_types: List of file types to concatenate (if None, auto-detect)
        save_format: Format to save combined files ('parquet', 'tsv', or 'both')
        
    Returns:
        dict: Mapping of file type to combined DataFrame
    """
    logger.info("=" * 60)
    logger.info("STEP 3: Concatenating files by type")
    logger.info("=" * 60)
    
    # Auto-detect file types if not specified
    if file_types is None:
        file_types = []
        for cluster_dir in output_dir.glob("cluster_*"):
            for type_dir in cluster_dir.iterdir():
                if type_dir.is_dir() and type_dir.name not in file_types:
                    file_types.append(type_dir.name)
        
        logger.info(f"Auto-detected file types: {file_types}")
    
    combined_dfs = {}
    
    for file_type in file_types:
        logger.info(f"\nProcessing {file_type} files...")
        df = concatenate_files_by_type(output_dir, file_type, save_format)
        
        if df is not None:
            combined_dfs[file_type] = df
    
    logger.info("=" * 60)
    logger.info(f"Concatenation complete: {len(combined_dfs)} file types processed")
    logger.info("=" * 60)
    
    for file_type, df in combined_dfs.items():
        logger.info(f"  {file_type}: {df.shape[0]:,} rows")
    
    return combined_dfs


if __name__ == "__main__":
    # valid_clusters = retrieve_valid_gh_analysis_clusters()
    valid_clusters = ["01"]  # for testing, only check the first two clusters
    
    owner = "emo-bon"
    
    # Define which files to download based on their "name" in ro-crate-metadata.json
    name_patterns = [
        "Tab-separated formatted taxon counts for LSU sequences",
        "Tab-separated formatted taxon counts for SSU sequences",
        # Add more patterns as needed:
        "Merged contigs GO summary",
        "Merged contigs InterProScan slim",
        "Merged contigs InterProScan",
        "Merged contigs KO summary",
        "Merged contigs PFAM summary"
    ]
    
    # Step 1: Clone all valid cluster repositories
    logger.info("=" * 60)
    logger.info("STEP 1: Cloning repositories")
    logger.info("=" * 60)
    
    cloned_repos = {}
    for cluster_id in valid_clusters:
        repo_name = f"analysis-results-cluster-{cluster_id}-crate"
        repo_path = clone_repository(owner, repo_name, TMP_DIR, force=False)
        if repo_path:
            cloned_repos[cluster_id] = repo_path
    
    logger.info(f"Successfully cloned {len(cloned_repos)} repositories")
    
    # Step 2: Find and process all ro-crate-metadata.json files
    logger.info("=" * 60)
    logger.info("STEP 2: Processing RO-Crate metadata files")
    logger.info("=" * 60)
    
    all_downloaded_files = []
    
    for cluster_id, repo_path in cloned_repos.items():
        logger.info(f"\nProcessing cluster {cluster_id}...")
        
        # Find all ro-crate-metadata.json files
        metadata_files = find_rocrate_metadata_files(repo_path)
        
        # Process each ro-crate folder
        for metadata_file in metadata_files[:3]:
            cluster_output_dir = OUTPUT_DIR / f"cluster_{cluster_id}"
            downloaded = process_rocrate_folder(metadata_file, cluster_output_dir, name_patterns)
            
            if downloaded:
                all_downloaded_files.append({
                    'cluster_id': cluster_id,
                    'rocrate': metadata_file.parent.name,
                    'files': downloaded
                })
    
    logger.info("=" * 60)
    logger.info(f"DOWNLOAD COMPLETE: Downloaded files from {len(all_downloaded_files)} RO-Crates")
    logger.info("=" * 60)
    
    # Summary
    for item in all_downloaded_files[:10]:  # Show first 10
        logger.info(f"Cluster {item['cluster_id']} - {item['rocrate']}: {list(item['files'].keys())}")
    
    if len(all_downloaded_files) > 10:
        logger.info(f"... and {len(all_downloaded_files) - 10} more")
    
    # Step 3: Concatenate all downloaded files by type
    concatenate_all_types(OUTPUT_DIR, save_format='parquet')
    
    # Step 4: Cleanup intermediate cluster folders
    logger.info("=" * 60)
    logger.info("STEP 4: Cleaning up intermediate files")
    logger.info("=" * 60)
    
    for cluster_dir in OUTPUT_DIR.glob("cluster_*"):
        try:
            logger.info(f"Removing {cluster_dir}...")
            shutil.rmtree(cluster_dir)
        except Exception as e:
            logger.error(f"Failed to remove {cluster_dir}: {e}")
    
    logger.info("Cleanup complete! Final parquet files saved in data/emo-bon_data/")
