# momics-demos
![Jupyterlab](https://img.shields.io/badge/Jupyter-notebook-brightgreen)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/palec87/momics-demos/)

Marine metagenomics platform NBs to get the science started. This work is part of [FAIR-EASE](https://fairease.eu/) project, specifically Pilot 5 for metagenomics to provide as many tools to for [emo-bon](https://data.emobon.embrc.eu/) data.

Please, consider open [issues](https://github.com/palec87/marine-omics/issues) with your dream workflow suggestions. I can be to certain extend your worker until 31/8. PRs are greatly welcome too.

# Design principles
1. Minimize dependencies to facilitate wide adaptation and further development of the codebase.
2. Simplicity over speed, however performance is considered.
3. Data import/export options after UDAL queries made easy.
4. Combining strengths of python/R/julia packages developed in those languages.
5. API calls to other services, such as Galaxy.

# Workflow notebooks
Notebooks always generate panel app for user friendly interactions. However working with the code using the same methods as the app should (needs to made sure of by testers) be straightforward.


## WF0, landing page app `not started`
General statistics of EMO-BON sequencing efforts


## WF1, Get and visualize some of the intermediate data products of the metaGOflow pipeline `pre-alpha`
Barebones in the `wf0_metagoflow` folder.


## WF2, Genetic diversity `alpha` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/palec87/momics-demos/blob/main/wf2_diversity/diversities_panel.ipynb)
NB provides visualization of alpha and beta diversities of the metaGOflow analyses. NB is located in `wf2_diversity/diversities_panel.ipynb`. Unfortunately I did not yet resolve hosting the dashboard properly on Colab.
 - Request access to the hosted version at the Blue cloud 2026 (BC) Virtual lab environment (VRE) [here](https://blue-cloud.d4science.org/).


## WF3-WF4, biosynthetic gene clusters (BGCs)
Work so far in the `gene_clusters` folder.


Your Galaxy access data should be stored as environmental variables in the `.env` file at the root of the repository
```
GALAXY_URL="https://earth-system.usegalaxy.eu/"
GALAXY_KEY="..."
```


### WF3, Running GECCO jobs on Galaxy  `pre-alpha`
1. Upload and run workflow.
2. Monitor the job.
3. Receive completion notification with some basic summary provided by Galaxy.


### WF4, Analyzing the BGCs `pre-pre-alpha`
1. Upload local data or query results of the GECCO from the Galaxy.
2. Identifying Biosynthetic Gene Clusters (BGCs).
3. Visualize BGCs.
4. Compare two samples in respect to each other.

## WF5, Integrate MGnify pipeline and data `not started`
1. Protein families comparison?

## WF6, R, Julia `not started`
1. Demonstrate usage of some relevant R and julia packages, Workflows
Q: Can it be done in a single NB? Should!

## WF7, DL package? `not started`
1. By the time, BC 2026 might have GPU support
2. Irrespective, try AI4EOSC perhaps? Q: Have not seen there much or any metagenomics though

## WF8, r-/k- communities `not started`
1. Correlate with Essential Ocean Variables (EOVs)

## WFX Some Visualization of some data `not started`
(This is probably WF0) Provides summary of up-to date statistics on amounts of sequenced and processed data.

# Dependencies
## General
- Currently `venv` is enough, no need for setting up `conda`, meaning that the dependencies are pure python.
- Utility functionalities are developed in parallel in this [repo](https://github.com/palec87/marine-omics). Currently not distributed with PyPI, install with `pip install https://github.com/palec87/marine-omics.git`.
- 
## Dashboards
- Dashboards are developed in [panel](https://panel.holoviz.org/)
  - If you put the NB code in the script, you can serve the dashboard in the browser using `panel serve app.py --dev`.
  - You can however serve the NB as well, `panel serve app.ipynb --dev`.
  - **Note**: if you want to run on Google Colab, you will need a `pyngrok` and ngrok token from [here](https://dashboard.ngrok.com/auth)
  - 
## Data
- For statistics, we use [pingouin](https://pingouin-stats.org/build/html/index.html) and [scikit-bio](https://scikit.bio/).
- Data part is handled by `pandas`, `numpy` etc. This might be upgraded to `polars`/`fire-ducks`.
- 
## Galaxy
- Galaxy support is built upon [bioblend](https://bioblend.readthedocs.io/en/latest/).

## Vizualization
- Visualization are currently **not** interactive and developed in `seaborn` or `matplotlib`. This will likely change to `holoviz`.
- Interactive parts use `jupyterlab`


