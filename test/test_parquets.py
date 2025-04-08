import pytest
import pandas as pd

@pytest.mark.parametrize(
    "table_name",
    [
        "go",
        "go_slim",
        "ips",
        "ko",
        "pfam",
        "SSU",
        "LSU",
    ],)

def test_parquet(table_name):
    folder = "data/parquet_files"
    data = pd.read_parquet(f"{folder}/metagoflow_analyses.{table_name}.parquet")   

    assert len(data) > 0

def test_column_names():
    folder = "data/parquet_files"
    data = pd.read_parquet(f"{folder}/metagoflow_analyses.go.parquet")
    expected = ["ref_code","id","name","aspect","abundance"]
    assert all([a == b for a, b in zip(data, expected)])

    data = pd.read_parquet(f"{folder}/metagoflow_analyses.go_slim.parquet")   
    assert all([a == b for a, b in zip(data, expected)])

    data = pd.read_parquet(f"{folder}/metagoflow_analyses.ips.parquet")
    expected = ["ref_code","accession","description","abundance"]
    assert all([a == b for a, b in zip(data, expected)])

    data = pd.read_parquet(f"{folder}/metagoflow_analyses.ko.parquet")
    expected = ["ref_code","entry","name","abundance"]
    assert all([a == b for a, b in zip(data, expected)])

    data = pd.read_parquet(f"{folder}/metagoflow_analyses.pfam.parquet")
    expected = ["ref_code","entry","name","abundance"]
    assert all([a == b for a, b in zip(data, expected)])

    data = pd.read_parquet(f"{folder}/metagoflow_analyses.SSU.parquet")
    expected = ["ref_code","ncbi_tax_id","abundance","superkingdom","kingdom","phylum","class","order","family","genus","species"]
    assert all([a == b for a, b in zip(data, expected)])
