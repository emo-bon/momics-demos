# momics-demos
Marine metagenomics platform NBs to get the science started.

# Dependencies
- Currently `venv` is enough, no need for setting up `conda`, meaning that the dependencies are pure python.
- Utility functionalities are developed in parallel in this [repo](https://github.com/palec87/marine-omics). Currently not distributed with PyPI, install with `pip install https://github.com/palec87/marine-omics.git`.
- Dashboards are developed in [panel](https://panel.holoviz.org/)
  - If you put the NB code in the script, you can serve the dashboard in the browser using `panel serve app.py --dev`.
  - You can however serve the NB as well, `panel serve app.ipynb --dev`.
- For statistics, we use [pingouin](https://pingouin-stats.org/build/html/index.html) and [scikit-bio](https://scikit.bio/).
- Data part is handled by `pandas`, `numpy` etc. This might be upgraded to `polars`.
- Visualization are currently not interactive and developed in `seaborn` or `matplotlib`. This will likely change.
- Interactive part uses `jupyterlab`.

# Design principles
1. Minimize dependencies to facilitate wide adaptation and further development of the codebase.
2. Simplicity over speed, however performance is considered.
3. Data export options after UDAL queries made easy.
4. Combining strengths of both python and R and packages developed in those languages
5. API calls to other services, such as Galaxy.

# Notebooks
## Visualization of the data
Provides summary of up-to date statistics on amounts of sequenced and processed data.

NB to be implemented

## Genetic diversity
NB provides visualization of alpha and beta diversities of the metaGOflow analyses. NB is located in `diversity/diversities_panel.ipynb`. Unfortunately I did not yet resolve hosting the dashboard properly on Colab.
 - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/palec87/momics-demos/blob/main/diversity/diversities_panel.ipynb)
 - Request access to the hosted version at the Blue cloud 2026 (BC) Virtual lab environment (VRE) [here](https://blue-cloud.d4science.org/).


## Gene Clusters
Your Galaxy access data should be stored as environmental variables in the `.env` file at the root of the repository
```
GALAXY_URL="https://earth-system.usegalaxy.eu/"
GALAXY_KEY="..."
```

Note: "https://usegalaxy.org/" cannot identify the GECCO tool in the toolshed, no idea why.

### Running GECCO jobs
1. Upload and run workflow.
2. Monitor the job.
3. Receive completion notification with some basic summary provided by Galaxy.

### Analyzing the BGCs
1. Upload local data or query results of the GECCO from the Galaxy.
2. Identifying Biosynthetic Gene Clusters (BGCs).
3. Visualize BGCs.
4. Compare two samples in respect to each other.
