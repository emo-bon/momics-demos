# momics-demos
![Jupyterlab](https://img.shields.io/badge/Jupyter-notebook-brightgreen)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/emo-bon/momics-demos/HEAD)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/palec87/momics-demos/)

Marine metagenomics platform NBs to get the science started. This work is part of [FAIR-EASE](https://fairease.eu/) project, specifically Pilot 5 for metagenomics to provide as many tools to for [emo-bon](https://data.emobon.embrc.eu/) data.

Please, consider opening [issues](https://github.com/palec87/marine-omics/issues) and PRs with your dream workflow suggestions. I can be to certain extend your worker until 31/8.


# Design principles
1. Minimize dependencies to facilitate wide adaptation and further development of the codebase.
2. Simplicity over speed, however performance is considered.
3. Data import/export options after UDAL queries made easy. (backend data queries developed by VLIZ)
4. Combining strengths of python/R/julia packages developed in those languages.
5. API calls to other services, such as Galaxy.


# Workflow notebooks
Notebooks always generate panel app for user friendly interactions. However working with the code using the same methods as the app should (needs to made sure of by testers) be straightforward.


## WF0, landing page app 
[![stability-alpha](https://img.shields.io/badge/stability-alpha-f4d03f.svg)](https://github.com/mkenney/software-guides/blob/master/STABILITY-BADGES.md#alpha) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/emo-bon/momics-demos/HEAD?urlpath=%2Fdoc%2Ftree%2Fwf0_landing_page%2Flanding_page.ipynb)

General statistics of EMO-BON sequencing efforts. The total amount of sampling events has reached more than a 1000 recently. Unfortunately `leafmap` widgets have problem with `ngrok` tunnels, so only binder integration is possible.


## WF1, Get and visualize some of the intermediate data products of the metaGOflow pipeline
[![stability-wip](https://img.shields.io/badge/stability-wip-lightgrey.svg)](https://github.com/mkenney/software-guides/blob/master/STABILITY-BADGES.md#work-in-progress)

Barebones in the `quality_control.ipynb` folder. There are almost 60 output files from the metaGOflow pipeline. This dashboard provides interface to most relevant intermediate ones, ie. all except from the taxonomy and functional analyses.


## WF2, Genetic diversity
[![stability-alpha](https://img.shields.io/badge/stability-alpha-f4d03f.svg)](https://github.com/mkenney/software-guides/blob/master/STABILITY-BADGES.md#alpha)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/emo-bon/momics-demos/HEAD?urlpath=%2Fdoc%2Ftree%2Fwf2_diversity%2Fdiversities_panel.ipynb)
 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/palec87/momics-demos/blob/main/wf2_diversity/diversities_panel.ipynb)

NB provides visualization of alpha and beta diversities of the metaGOflow analyses. NB is located in `diversities_panel.ipynb`. Unfortunately I did not yet resolve hosting the dashboard properly on Colab.
 - Request access to the hosted version at the Blue cloud 2026 (BC) Virtual lab environment (VRE) [here](https://blue-cloud.d4science.org/).

**ADVANCED diversity dashboard** (heavier, but contains pivot tables on taxonomy LSU and SSU tables). Next step is to provide permanova calculations and migrate to seaborn plots.
[![stability-alpha](https://img.shields.io/badge/stability-alpha-f4d03f.svg)](https://github.com/mkenney/software-guides/blob/master/STABILITY-BADGES.md#alpha)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/emo-bon/momics-demos/HEAD?urlpath=%2Fdoc%2Ftree%2Fwf2_diversity%2Fdiversities_panel_advanced.ipynb)


## WF3-WF4, biosynthetic gene clusters (BGCs)
You will need an account on the galaxy [earth-system](https://earth-system.usegalaxy.eu/) for this NBs to work. Your Galaxy access data should be stored as environmental variables in the `.env` file at the root of the repository
```
GALAXY_URL="https://earth-system.usegalaxy.eu/"
GALAXY_KEY="..."
```

### WF3, Running GECCO jobs on Galaxy 
[![stability-alpha](https://img.shields.io/badge/stability-alpha-f4d03f.svg)](https://github.com/mkenney/software-guides/blob/master/STABILITY-BADGES.md#alpha)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/emo-bon/momics-demos/HEAD?urlpath=%2Fdoc%2Ftree%2Fwf3_gene_clusters%2Fbgc_run_gecco_job.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/palec87/momics-demos/blob/main/wf3_gene_clusters/bgc_run_gecco_job.ipynb)

BUG: For unknown reason the Binder dashboard does not work.
Dashboard illustrating submission of jobs to galaxy (GECCO tool) in `bgc_run_gecco.ipynb`.

1. Upload and run workflow.
2. Or start the workflow with existing data and in existing history on Galaxy.
3. Monitor the job.


### WF4, Analyzing the BGCs
[![stability-wip](https://img.shields.io/badge/stability-wip-lightgrey.svg)](https://github.com/mkenney/software-guides/blob/master/STABILITY-BADGES.md#work-in-progress) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/palec87/momics-demos/blob/main/wf3_gene_clusters/bgc_analyze_gecco.ipynb)

1. Upload local data or query results of the GECCO from the Galaxy.
2. Identifying Biosynthetic Gene Clusters (BGCs).
3. Visualize BGCs.
4. Compare two samples in respect to each other.


## WF5, Integrate MGnify pipeline and data `in-progress`
[![stability-wip](https://img.shields.io/badge/stability-wip-lightgrey.svg)](https://github.com/mkenney/software-guides/blob/master/STABILITY-BADGES.md#work-in-progress) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/palec87/momics-demos/blob/main/wf5_MGnify/query_data.ipynb) 

*dependencies not yet fixed*

The examples are heavily inspired and taken from the MGnify project [itself](https://github.com/EBI-Metagenomics/notebooks/tree/main/src/notebooks)

1. How to query data and make basic plots such as Sankey from the MGnify database `wf5_MGnify/query_data.ipynb` 
2. Protein families comparison?


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
- Utility functionalities are developed in parallel in this [repo](https://github.com/emo-bon/marine-omics-methods). Currently not distributed with PyPI, install with `pip install https://github.com/emo-bon/marine-omics-methods.git`.


## Dashboards
- Dashboards are developed in [panel](https://panel.holoviz.org/)
  - If you put the NB code in the script, you can serve the dashboard in the browser using `panel serve app.py --dev`.
  - You can however serve the NB as well, `panel serve app.ipynb --dev`.
  - **Note**: if you want to run on Google Colab, you will need a `pyngrok` and ngrok token from [here](https://dashboard.ngrok.com/auth)
  - `Binder` integration is better in terms of running dashboards, but loading the repo might take time or crash, so `GColab` in that case is a better option.


## Data
- For statistics, we use [pingouin](https://pingouin-stats.org/build/html/index.html) and [scikit-bio](https://scikit.bio/).
- Data part is handled by `pandas`, `numpy` etc. This might be upgraded to `polars`/`fire-ducks`.


## Galaxy
- Galaxy support is built upon [bioblend](https://bioblend.readthedocs.io/en/latest/).


## Vizualization
- Visualization are currently **not** interactive and developed in `seaborn` or `matplotlib`. This will likely change to `holoviz`.
- Interactive parts use `jupyterlab`


