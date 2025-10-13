import pytest
import nbformat
import os
from nbclient import NotebookClient

# list of notebooks you want to test
NOTEBOOKS = [
    "wf0_landing_page/landing_page.ipynb",
    "wf0_landing_page/landing_page_interactive.ipynb",
    # "wf1_metagoflow/quality_control.ipynb",  # used to pass
    # "wf1_metagoflow/quality_control_interactive.ipynb",  # used to pass
    "wf2_diversity/diversities_panel.ipynb",
    # "wf2_diversity/diversities_panel_interactive.ipynb",  # because selection need to be made
    "wf2_diversity/diversities_panel_advanced.ipynb",
    # "wf2_diversity/diversities_panel_advanced_interactive.ipynb"  # because selection need to be made
    # "wf3_gene_clusters/bgc_analyze_gecco_single.ipynb",  # needs authentication
    # "wf3_gene_clusters/bgc_analyze_gecco_single_interactive.ipynb",  # because selection need to be made
    # "wf3_gene_clusters/bgc_run_gecco_job.ipynb",  # missing access to the galaxy login
    # "wf3_gene_clusters/bgc_run_gecco_job_interactive.ipynb",  # missing access to the galaxy login
    "wf4_co-occurrence/parametrized_taxonomy.ipynb",
    # "wf4_co-occurrence/parametrized_taxonomy_interactive.ipynb"  # not sure about this error
    "wf5_MGnify/01_summary_analysis_panel_wip.ipynb",
    "wf5_MGnify/02_comparative_taxonomy.ipynb",
    "wfs_extra/pathogen_taxonomy.ipynb"

]

@pytest.mark.parametrize("nb_path", NOTEBOOKS)
def test_notebook_runs(nb_path):
    """Test that a notebook runs without error."""
    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute(cwd=os.path.dirname(nb_path))
