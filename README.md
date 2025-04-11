<div id="top"></div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <img src="docs/1_workflow.png" alt="Workflow diagram">

  <h1 align="center">GoiStrat</h1>
  <h4 align="center">Gene-of-interest-based sample stratification for the evaluation of functional differences</h4>

</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#data">Data</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#usage">Usage</a></li>
        <li><a href="#file-descriptions">File descriptions</a></li>
      </ul>
    </li>
    <li><a href="#project-documentation">Project Documentation</a></li>
    <li><a href="#tutorial-performing-sample-stratification-with-goistrat-and-the-topbottom-approach-plus-differential-analyses-on-multiple-datasets">Tutorial: Performing sample stratification with GoiStrat and the top/bottom approach, plus differential analyses, on multiple datasets</a></li>
    <li><a href="#tutorial-downstream-analyses-on-the-folh1-use-case">Tutorial: Downstream analyses on the FOLH1 use case</a></li>
    <li><a href="#additional-notes">Additional Notes</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
    <li><a href="#citation">Citation</a></li>
  </ol>
</details>

## About The Project

This repository contains the implementation of *GoiStrat - Gene-of-interest-based sample stratification for the evaluation of functional differences*. See [publication](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-025-06109-0) for details.

The implementation was done entirely in Python, using [*rpy2*](https://github.com/rpy2/rpy2) wrappers for all necessary R packages.

### Data

This workflow was tested by applying it to *FOLH1* in prostate cancer. The data sources used were:

- The Prostate Cancer Transcriptome Atlas (PCTA) [[Paper](https://www.nature.com/articles/s41467-021-26840-5)].
- The Cancer Genome Atlas Prostate Adenocarcinoma (TCGA-PRAD) | [[Paper](https://pubmed.ncbi.nlm.nih.gov/26544944/)].
- The West Coast Prostate Cancer Dream Team - Metastatic Castration Resistant Prostate Cancer (WCDT-MCRPC) | [[Paper 1](https://pubmed.ncbi.nlm.nih.gov/30033370/), [Paper 2](https://pubmed.ncbi.nlm.nih.gov/33077885/)].

If you wish you reproduce the results shown in the paper, you must obtain the permissions from the owners of the data when required. Downloading and processing of the data is also a pre-requisite. Helper functions and scripts under `src/pipelines/data` were used for such purposes.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

### Prerequisites

You should have an Anaconda environment installed in your UNIX system (currently only Ubuntu/CentOS has been tested). I recommend using `Miniforge3`:

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

More detailed instructions to setup Anaconda using Miniforge3 can be found [in their GitHub page](https://github.com/conda-forge/miniforge).

### Installation

Here's a step-by-step guide to setup the library:

1. Clone this repository:

  ```bash
  git clone https://github.com/CarlosUziel/goi-strat
  ```

2. Install mamba:

```bash
conda install -n base -c conda-forge mamba
```

3. Create virtual environment:

*Option 1:*
```bash
bash goi-strat/setup_env.sh
```

**Note**: Please keep in mind that `setup_env.sh` might include unnecessary dependencies, as this was a general environment we used for all our projects. You are invited to remove and add dependencies as you see fit.

*Option 2:*
```bash
mamba env create -f environment.yml # alternatively try environment_hist.yml
mamba activate bioinfo
```

This will take a while, have patience.

4. Set `PYTHONPATH` variable (preferably in your `.bashrc` file or equivalent):

```bash
export PYTHONPATH="/home/{user}/goi-strat/src":$PYTHONPATH
```

Now you are ready to start using the **GoiStrat** workflow!

<!-- USAGE EXAMPLES -->

### Usage

While this library contains plenty of ready-to-use scripts to build complete pipelines, it also provides multiple utility functions that can be used individually as per user needs. A special effort has been put into making the design of the library highly modular and flexible for different use cases. Feel free to submerge yourself in the codebase.

The entire workflow (except the processing of FASTQ files) is described in `src/pipelines/psma_levels_workflow.sh`. If you need to process FASTQ files to obtain DNA methylation signatures, please refer to the scripts under `src/pipelines/fastq_processing`.

### File descriptions

This repository is organized as follows:

- `docs`: Files of the main workflow figure.
- `notebooks/paper-figures.ipynb`: Jupyter notebook to reproduce all figures shown in the paper.
- `src/components`: Component classes used in various pipelines, mainly for functional analysis.
- `src/data`: Helper functions for data processing.
- `src/pipelines`: Multiple individual scripts, each representing a step in the workflow. See details below for each step (in alphabetical order):
  - `src/pipelines/data`: Downloads data, its annotations and applies some minor pre-processing steps. Relevant R packages used here include **DESeq2**, **msigdbr** and **TCGAbiolinks**.
  - `src/pipelines/degss_genes_stats`: Obtains gene statistics from differentially enriched gene sets (DEGSs).
  - `src/pipelines/degss_ppi_networks`: Generates PPI networks from the genes within DEGSs. PPI relationships are extracted from *STRINGDB* using the R package **rbioapi**.
  - `src/pipelines/degss_ppi_networks_clustering`: Clusters proteins in the PPI networks using **Node2Vec** embeddings and ensemble clustering.
  - `src/pipelines/differential_enrichment`: Runs differential enrichment analyses with **limma**, obtaining DEGSs in the process.
  - `src/pipelines/differential_expression`: Runs differential expression analyses with **DESeq2**, obtaining differentially expressed genes (DEGs) in the process.
  - `src/pipelines/differential_methylation`: Runs differential methylation analyses with **minfi** (DNA Methylation array) and **methylkit** (RRBS), obtaining differentially methylated genes (DMRs) in the process.
  - `src/pipelines/fastq_processing`: Processes raw FASTQ files (WCDT-MCRPC dataset only). It includes quality control with **fasqc** and **multiqc**, adapter trimming with **cutadapt** and/or **trim-galore** and mapping using **bismark**, also used to extract methylation values. It is worth mentioning that while only the scripts to process WCDT-MCRPC are included, the `utils.py` file includes wrappers for many additional tools that might prove useful for other datasets, such as **STAR**, **bowtie2**, **samtools** and others.
  - `src/pipelines/functional_analysis`: Runs functional analyses (i.e. GSEA, ORA) on many gene sets collections (e.g. MSigDB H and C1-C8, DO, GO, Reactome, KEGG, MKEGG and NCG), obtaining enriched gene sets in the process. Relevant R packages used here include **clusterProfiler**, **dose**, **enrichplot**, **pathview** and **AnnotationHub**.
  - `src/pipelines/integrative_analysis`: Combines enriched gene sets from different methods and datasets (i.e. Gene sets from D-GSVA, GSEA on DEGS and GSEA on DMRs).
- `src/r_wrappers`: Python wrappers for all the underlying R packages used in the pipelines.
- `slurm`: Utility functions to run scripts on a SLURM system.

<br>

> This repository is an extract of all the code I developed during my PhD. While care has been taken to remove anything unrelated to `GoiStrat`, some artifacts might still be present. In any case, they can be safely ignored.

<p align="right">(<a href="#top">back to top</a>)</p>

## Project Documentation

The documentation for the GoiStrat workflow entire codebase is generated using Sphinx. To build the documentation, navigate to the `docs` directory and run the following command:

```bash
make html
```

This will generate the HTML documentation in the `docs/build` directory. Open the `index.html` file in your browser to view the documentation or run the following command to open it in your default browser:

```bash
open build/html/index.html
```

<p align="right">(<a href="#top">back to top</a>)</p>


## Tutorial: Performing sample stratification with GoiStrat and the top/bottom approach, plus differential analyses, on multiple datasets

This tutorial will guide you through the first phase of the GoiStrat workflow, which involves sample stratification, as well as the first part of the second phase. We will follow the scripts outlined `src/pipelines/multi_dataset_workflow.sh`.

### 1. Download, Process, and Split Data

The first step involves downloading, processing, and splitting the data. This is done using the following scripts:

#### `src/pipelines/data/run/rna_seq/multi_dataset/download_data.py`:

This script downloads the necessary RNA-Seq data for multiple TCGA datasets. It ensures that the data is properly formatted and stored in the appropriate directories for further processing.

**Global Variables (user-provided):**
- `STORAGE`: The root directory where all data will be stored.
- `TCGA_NAMES`: Names of the TCGA datasets to download.

#### `src/pipelines/data/run/rna_seq/multi_dataset/generate_gsva.py`:

This script generates Gene Set Variation Analysis (GSVA) scores for the downloaded RNA-Seq data. GSVA is used to estimate variation of gene set enrichment through the samples of an expression dataset.

**Global Variables (user-provided):**
- `STORAGE`: The root directory where all data will be stored.
- `DATASET_NAMES`: The names of the datasets being processed, which should already be downloaded.
- `SAMPLE_CONTRAST_FACTOR`: The factor used to contrast different sample types.
- `CONTRASTS_LEVELS`: The levels of the contrast factor being analysed.
- `CONTRASTS_LEVELS_COLORS`: A dictionary mapping contrast levels to colors.
- `SPECIES`: The species for which the data is being processed (e.g. *Homo sapiens*).
- `MSIGDB_CATS`: The collections of MSigDB from which to extract gene sets to build the GSVA matrices.
- `PARALLEL`: A flag indicating whether to run the analysis in parallel, for each dataset and MSigDB collection.

#### `src/pipelines/data/run/rna_seq/multi_dataset/goi_gsva_splits_annotation.py`:

This script splits the samples based on the GoiStrat sliding-window algorithm with functional difference scores (FDS). It also generates the corresponding annotation file marking which group each sample belongs to.

**Global Variables (user-provided):**
- `STORAGE`: The root directory for loading/storing data.
- `SPECIES`: The species name, set to "Homo sapiens".
- `P_COLS`: An iterable of p-value columns used for analysis.
- `P_THS`: An iterable of p-value thresholds used for analysis.
- `LFC_LEVELS`: An iterable of log fold change levels used for analysis.
- `LFC_THS`: An iterable of log fold change thresholds used for analysis.
- `MSIGDB_CATS`: An iterable of MSigDB categories used for analysis.
- `MIN_PERCENTILE`: The minimum percentile used for group distribution.
- `MID_PERCENTILE`: The midpoint percentile used for group distribution.
- `PARALLEL`: A boolean indicating whether to run processes in parallel.
- `DATASETS_MARKERS`: A dictionary mapping dataset names to their respective gene markers.
- `GOI_SYMBOL`: The symbol of the gene of interest.
- `GOI_ENSEMBL`: The Ensembl ID of the gene of interest.
- `MAIN_ROOT`: The main root path for a dataset.
- `ANNOT_PATH`: The path to the samples annotation CSV file.
- `RAW_COUNTS_PATH`: The path to the raw counts CSV file.
- `GSVA_PATH`: The path to the GSVA data directory.
- `DATA_ROOT`: The root path for loading/storing data related to a specific dataset and gene of interest.
- `RESULTS_PATH`: The path for loading/storing results.
- `DATA_PATH`: The path for loading/storing data files.
- `ANNOT_PATH_NEW`: The path for the new annotation CSV file.
- `SAMPLE_CONTRAST_FACTOR`: The factor used to contrast different sample types.
- `CONTRASTS_LEVELS`: An iterable of contrast levels.

#### `src/pipelines/data/run/rna_seq/multi_dataset/goi_perc_splits_annotation.py`:

This script annotates the samples based on top/bottom strategy for different quantiles (i.e. percentages) for the gene of interest (GOI).

**Global Variables (user-provided):**
- `STORAGE`: The root directory for loading/storing data.
- `SPECIES`: The species name, set to "Homo sapiens".
- `PARALLEL`: A boolean indicating whether to run processes in parallel.
- `CONTRASTS_LEVELS_COLORS`: A dictionary mapping contrast levels to their respective colors.
- `GOI_LEVELS_COLORS`: A dictionary mapping gene of interest levels to their respective colors.
- `DATASETS_MARKERS`: A dictionary mapping dataset names to their respective gene markers.
- `PERCENTILES`: An iterable of percentiles used for analysis.
- `GOI_SYMBOL`: The symbol of the gene of interest.
- `GOI_ENSEMBL`: The Ensembl ID of the gene of interest.
- `MAIN_ROOT`: The main root path for a dataset.
- `ANNOT_PATH`: The path to the samples annotation CSV file.
- `RAW_COUNTS_PATH`: The path to the raw counts CSV file.
- `DATA_ROOT`: The root path for loading/storing data related to a specific dataset and gene of interest.
- `DATA_PATH`: The path for loading/storing data files.
- `PLOTS_PATH`: The path for loading/storing plot files.
- `ANNOT_PATH_NEW`: The path for the new annotation CSV file.
- `SAMPLE_CONTRAST_FACTOR`: The factor used to contrast different sample types.
- `CONTRASTS_LEVELS`: An iterable of contrast levels.

<p align="right">(<a href="#top">back to top</a>)</p>

### 2. Differential Analyses

The second step involves running differential analyses on the stratified samples. This is done using the following scripts:

#### `src/pipelines/differential_expression/run/multi_dataset/goi_gsva.py`:

This script performs differential expression analysis on the GoiStrat splits. It identifies differentially expressed genes based on the stratified samples.

**Global Variables (user-provided):**
- `STORAGE`: The root directory for loading/storing data.
- `SPECIES`: The species name, set to "Homo sapiens".
- `P_COLS`: An iterable of p-value columns used for analysis.
- `P_THS`: An iterable of p-value thresholds used for analysis.
- `LFC_LEVELS`: An iterable of log fold change levels used for analysis.
- `LFC_THS`: An iterable of log fold change thresholds used for analysis.
- `HEATMAP_TOP_N`: The number of top genes to include in the heatmap.
- `COMPUTE_VST`: A boolean indicating whether to compute variance stabilizing transformation.
- `COMPUTE_RLOG`: A boolean indicating whether to compute regularized log transformation.
- `PARALLEL`: A boolean indicating whether to run processes in parallel.
- `CONTRASTS_LEVELS_COLORS`: A dictionary mapping contrast levels to their respective colors.
- `GOI_LEVELS_COLORS`: A dictionary mapping gene of interest levels to their respective colors.
- `DATASETS_MARKERS`: A dictionary mapping dataset names to their respective gene markers.
- `GOI_SYMBOL`: The symbol of the gene of interest.
- `GOI_ENSEMBL`: The Ensembl ID of the gene of interest.
- `MAIN_ROOT`: The main root path for a dataset.
- `DATA_ROOT`: The root path for loading/storing data related to a specific dataset and gene of interest.
- `RESULTS_PATH`: The path for loading/storing results.
- `PLOTS_PATH`: The path for loading/storing plot files.
- `DATA_PATH`: The path for loading/storing data files.
- `ANNOT_PATH`: The path to the samples annotation CSV file.
- `RAW_COUNTS_PATH`: The path to the raw counts CSV file.
- `SAMPLE_CONTRAST_FACTOR`: The factor used to contrast different sample types.
- `GOI_LEVEL_PREFIX`: The prefix used for the levels of the gene of interest.
- `GOI_CLASS_PREFIX`: The prefix used for the classes of the gene of interest.
- `SAMPLE_CLUSTER_CONTRAST_LEVELS`: An iterable of sample cluster contrast levels.

#### `src/pipelines/differential_expression/run/multi_dataset/goi_perc.py`:

This script performs differential expression analysis on the GoiStrat splits. It identifies differentially expressed genes based on the stratified samples.

**Global Variables (user-provided):**
- `STORAGE`: The root directory for loading/storing data.
- `SPECIES`: The species name, set to "Homo sapiens".
- `P_COLS`: An iterable of p-value columns used for analysis.
- `P_THS`: An iterable of p-value thresholds used for analysis.
- `LFC_LEVELS`: An iterable of log fold change levels used for analysis.
- `LFC_THS`: An iterable of log fold change thresholds used for analysis.
- `HEATMAP_TOP_N`: The number of top genes to include in the heatmap.
- `COMPUTE_VST`: A boolean indicating whether to compute variance stabilizing transformation.
- `COMPUTE_RLOG`: A boolean indicating whether to compute regularized log transformation.
- `PARALLEL`: A boolean indicating whether to run processes in parallel.
- `CONTRASTS_LEVELS_COLORS`: A dictionary mapping contrast levels to their respective colors.
- `GOI_LEVELS_COLORS`: A dictionary mapping gene of interest levels to their respective colors.
- `DATASETS_MARKERS`: A dictionary mapping dataset names to their respective gene markers.
- `PERCENTILES`: An iterable of percentiles used for analysis.
- `GOI_SYMBOL`: The symbol of the gene of interest.
- `GOI_ENSEMBL`: The Ensembl ID of the gene of interest.
- `MAIN_ROOT`: The main root path for a dataset.
- `DATA_ROOT`: The root path for loading/storing data related to a specific dataset and gene of interest.
- `RESULTS_PATH`: The path for loading/storing results.
- `PLOTS_PATH`: The path for loading/storing plot files.
- `DATA_PATH`: The path for loading/storing data files.
- `ANNOT_PATH`: The path to the samples annotation CSV file.
- `RAW_COUNTS_PATH`: The path to the raw counts CSV file.
- `SAMPLE_CONTRAST_FACTOR`: The factor used to contrast different sample types.
- `GOI_LEVEL_PREFIX`: The prefix used for the levels of the gene of interest.
- `GOI_CLASS_PREFIX`: The prefix used for the classes of the gene of interest.
- `SAMPLE_CLUSTER_CONTRAST_LEVELS`: An iterable of sample cluster contrast levels.

#### `src/pipelines/differential_enrichment/run/multi_dataset/goi_gsva.py`:

This script performs differential enrichment analysis using GSVA scores. It identifies differentially enriched gene sets based on the stratified samples.

**Global Variables (user-provided):**
- `STORAGE`: The root directory for loading/storing data.
- `SPECIES`: The species name, set to "Homo sapiens".
- `P_COLS`: An iterable of p-value columns used for analysis.
- `P_THS`: An iterable of p-value thresholds used for analysis.
- `LFC_LEVELS`: An iterable of log fold change levels used for analysis.
- `LFC_THS`: An iterable of log fold change thresholds used for analysis.
- `HEATMAP_TOP_N`: The number of top genes to include in the heatmap.
- `MSIGDB_CATS`: An iterable of MSigDB categories used for analysis.
- `PARALLEL`: A boolean indicating whether to run processes in parallel.
- `CONTRASTS_LEVELS_COLORS`: A dictionary mapping contrast levels to their respective colors.
- `GOI_LEVELS_COLORS`: A dictionary mapping gene of interest levels to their respective colors.
- `DATASETS_MARKERS`: A dictionary mapping dataset names to their respective gene markers.
- `GOI_SYMBOL`: The symbol of the gene of interest.
- `GOI_ENSEMBL`: The Ensembl ID of the gene of interest.
- `MAIN_ROOT`: The main root path for a dataset.
- `DATA_ROOT`: The root path for loading/storing data related to a specific dataset and gene of interest.
- `RESULTS_PATH`: The path for loading/storing results.
- `PLOTS_PATH`: The path for loading/storing plot files.
- `DATA_PATH`: The path for loading/storing data files.
- `ANNOT_PATH`: The path to the samples annotation CSV file.
- `GSVA_PATH`: The path to the GSVA data directory.
- `SAMPLE_CONTRAST_FACTOR`: The factor used to contrast different sample types.
- `GOI_LEVEL_PREFIX`: The prefix used for the levels of the gene of interest.
- `GOI_CLASS_PREFIX`: The prefix used for the classes of the gene of interest.
- `SAMPLE_CLUSTER_CONTRAST_LEVELS`: An iterable of sample cluster contrast levels.

#### `src/pipelines/differential_enrichment/run/multi_dataset/goi_perc.py`:

This script performs differential enrichment analysis on the top/bottom splits. It identifies differentially enriched gene sets based on the stratified samples.

**Global Variables (user-provided):**
- `STORAGE`: The root directory for loading/storing data.
- `SPECIES`: The species name, set to "Homo sapiens".
- `P_COLS`: An iterable of p-value columns used for analysis.
- `P_THS`: An iterable of p-value thresholds used for analysis.
- `LFC_LEVELS`: An iterable of log fold change levels used for analysis.
- `LFC_THS`: An iterable of log fold change thresholds used for analysis.
- `HEATMAP_TOP_N`: The number of top genes to include in the heatmap.
- `MSIGDB_CATS`: An iterable of MSigDB categories used for analysis.
- `PARALLEL`: A boolean indicating whether to run processes in parallel.
- `CONTRASTS_LEVELS_COLORS`: A dictionary mapping contrast levels to their respective colors.
- `GOI_LEVELS_COLORS`: A dictionary mapping gene of interest levels to their respective colors.
- `DATASETS_MARKERS`: A dictionary mapping dataset names to their respective gene markers.
- `PERCENTILES`: An iterable of percentiles used for analysis.
- `GOI_SYMBOL`: The symbol of the gene of interest.
- `GOI_ENSEMBL`: The Ensembl ID of the gene of interest.
- `MAIN_ROOT`: The main root path for a dataset.
- `DATA_ROOT`: The root path for loading/storing data related to a specific dataset and gene of interest.
- `RESULTS_PATH`: The path for loading/storing results.
- `PLOTS_PATH`: The path for loading/storing plot files.
- `DATA_PATH`: The path for loading/storing data files.
- `ANNOT_PATH`: The path to the samples annotation CSV file.
- `GSVA_PATH`: The path to the GSVA data directory.
- `SAMPLE_CONTRAST_FACTOR`: The factor used to contrast different sample types.
- `GOI_LEVEL_PREFIX`: The prefix used for the levels of the gene of interest.
- `GOI_CLASS_PREFIX`: The prefix used for the classes of the gene of interest.
- `SAMPLE_CLUSTER_CONTRAST_LEVELS`: An iterable of sample cluster contrast levels.

<p align="right">(<a href="#top">back to top</a>)</p>

## Tutorial: Downstream analyses on the *FOLH1* use case
This tutorial will guide you through the second phase of the GoiStrat workflow, after differential analyses. We will follow the scripts outlined the last section of`src/pipelines/psma_levels_workflow.sh.sh`.

### `src/pipelines/integrative_analysis/run/rna_seq/pcta_wcdt/goi_intersect_degss_gsea.py`:

This script combines enriched gene sets from different methods and datasets (i.e., Gene sets from D-GSVA, GSEA on DEGS, and GSEA on DMRs).

**Global Variables (user-provided):**
- `STORAGE`: The root directory for loading/storing data.
- `GOI_ENSEMBL`: The Ensembl ID for the gene of interest (FOLH1/PSMA).
- `SPECIES`: The species name, set to "Homo sapiens".
- `GOI_SYMBOL`: The symbol of the gene of interest.
- `DATA_ROOT`: The root path for loading/storing data related to a specific dataset and gene of interest.
- `PRIM_METH_ROOT`: The root path for loading/storing primary methylation data.
- `MET_METH_ROOT`: The root path for loading/storing metastatic methylation data.
- `DATA_PATH`: The path for loading/storing data files.
- `ANNOT_PATH`: The path to the samples annotation CSV file.
- `SAMPLE_CONTRAST_FACTOR`: The factor used to contrast different sample types.
- `GOI_LEVEL_PREFIX`: The prefix used for the levels of the gene of interest.
- `CONTRAST_COMPARISONS`: A dictionary mapping contrast levels to their respective filters.
- `P_COLS`: An iterable of p-value columns used for analysis.
- `P_THS`: An iterable of p-value thresholds used for analysis.
- `LFC_LEVELS`: An iterable of log fold change levels used for analysis.
- `LFC_THS`: An iterable of log fold change thresholds used for analysis.
- `MSIGDB_CATS`: An iterable of MSigDB categories used for analysis.
- `PARALLEL`: A boolean indicating whether to run processes in parallel.

### `src/pipelines/degss_genes_stats/run/rna_seq/pcta_wcdt/goi.py`:

This script obtains gene statistics from differentially enriched gene sets (DEGSS), mainly z-score normalised gene occurrences.

**Global Variables (user-provided):**
- `STORAGE`: The root directory for loading/storing data.
- `GOI_ENSEMBL`: The Ensembl ID for the gene of interest (FOLH1/PSMA).
- `SPECIES`: The species name, set to "Homo sapiens".
- `GOI_SYMBOL`: The symbol of the gene of interest.
- `DATA_ROOT`: The root path for loading/storing data related to a specific dataset and gene of interest.
- `PRIM_METH_ROOT`: The root path for loading/storing primary methylation data.
- `MET_METH_ROOT`: The root path for loading/storing metastatic methylation data.
- `DATA_PATH`: The path for loading/storing data files.
- `GSVA_PATH`: The path to the GSVA data directory.
- `INT_ANALYSIS_PATH`: The path for loading/storing integrative analysis data.
- `GENES_STATS_PATH`: The path for loading/storing gene statistics data.
- `ANNOT_PATH`: The path to the samples annotation CSV file.
- `SAMPLE_CONTRAST_FACTOR`: The factor used to contrast different sample types.
- `GOI_LEVEL_PREFIX`: The prefix used for the levels of the gene of interest.
- `CONTRAST_COMPARISONS`: A dictionary mapping contrast levels to their respective filters.
- `P_COLS`: An iterable of p-value columns used for analysis.
- `P_THS`: An iterable of p-value thresholds used for analysis.
- `LFC_LEVELS`: An iterable of log fold change levels used for analysis.
- `LFC_THS`: An iterable of log fold change thresholds used for analysis.
- `MSIGDB_CATS`: An iterable of MSigDB categories used for analysis.
- `BOOTSTRAP_ITERATIONS`: The number of bootstrap iterations for statistical analysis.

### `src/pipelines/degss_genes_stats/run/rna_seq/pcta_wcdt/goi_metadata.py`:

This script gathers all metadata for the genes within DEGSS.

**Global Variables (user-provided):**
- `STORAGE`: The root directory for loading/storing data.
- `GOI_ENSEMBL`: The Ensembl ID for the gene of interest (FOLH1/PSMA).
- `SPECIES`: The species name, set to "Homo sapiens".
- `GOI_SYMBOL`: The symbol of the gene of interest.
- `DATA_ROOT`: The root path for loading/storing data related to a specific dataset and gene of interest.
- `PRIM_METH_ROOT`: The root path for loading/storing primary methylation data.
- `MET_METH_ROOT`: The root path for loading/storing metastatic methylation data.
- `DATA_PATH`: The path for loading/storing data files.
- `GSVA_PATH`: The path to the GSVA data directory.
- `INT_ANALYSIS_PATH`: The path for loading/storing integrative analysis data.
- `GENES_STATS_PATH`: The path for loading/storing gene statistics data.
- `ANNOT_PATH`: The path to the samples annotation CSV file.
- `SAMPLE_CONTRAST_FACTOR`: The factor used to contrast different sample types.
- `GOI_LEVEL_PREFIX`: The prefix used for the levels of the gene of interest.
- `CONTRAST_COMPARISONS`: A dictionary mapping contrast levels to their respective filters.
- `P_COLS`: An iterable of p-value columns used for analysis.
- `P_THS`: An iterable of p-value thresholds used for analysis.
- `LFC_LEVELS`: An iterable of log fold change levels used for analysis.
- `LFC_THS`: An iterable of log fold change thresholds used for analysis.
- `MSIGDB_CATS`: An iterable of MSigDB categories used for analysis.

### `src/pipelines/degss_ppi_networks/run/rna_seq/pcta_wcdt/goi.py`:

This script generates PPI networks from the genes (filtered by significant z-scored normalised gene occurrences) within DEGSS. PPI relationships are extracted from *STRINGDB* using the R package **rbioapi**.

**Global Variables (user-provided):**
- `STORAGE`: The root directory for storing data.
- `GOI_ENSEMBL`: The Ensembl ID for the gene of interest (FOLH1/PSMA).
- `SPECIES`: The species name, set to "Homo sapiens".
- `GOI_SYMBOL`: The symbol of the gene of interest.
- `DATA_ROOT`: The root path for storing data related to a specific dataset and gene of interest.
- `PRIM_METH_ROOT`: The root path for storing primary methylation data.
- `MET_METH_ROOT`: The root path for storing metastatic methylation data.
- `DATA_PATH`: The path for storing data files.
- `GSVA_PATH`: The path to the GSVA data directory.
- `INT_ANALYSIS_PATH`: The path for storing integrative analysis data.
- `GENES_STATS_PATH`: The path for storing gene statistics data.
- `ANNOT_PATH`: The path to the samples annotation CSV file.
- `SAMPLE_CONTRAST_FACTOR`: The factor used to contrast different sample types.
- `GOI_LEVEL_PREFIX`: The prefix used for the levels of the gene of interest.
- `CONTRAST_COMPARISONS`: A dictionary mapping contrast levels to their respective filters.
- `P_COLS`: An iterable of p-value columns used for analysis.
- `P_THS`: An iterable of p-value thresholds used for analysis.
- `LFC_LEVELS`: An iterable of log fold change levels used for analysis.
- `LFC_THS`: An iterable of log fold change thresholds used for analysis.
- `MSIGDB_CATS`: An iterable of MSigDB categories used for analysis.

### `src/pipelines/degss_ppi_networks_clustering/run/rna_seq/pcta_wcdt/goi.py`:

This script clusters proteins in the PPI networks using **Node2Vec** embeddings and ensemble clustering.

**Global Variables (user-provided):**
- `STORAGE`: The root directory for storing data.
- `GOI_ENSEMBL`: The Ensembl ID for the gene of interest (FOLH1/PSMA).
- `SPECIES`: The species name, set to "Homo sapiens".
- `GOI_SYMBOL`: The symbol of the gene of interest.
- `DATA_ROOT`: The root path for storing data related to a specific dataset and gene of interest.
- `DATA_PATH`: The path for storing data files.
- `PPI_NETWORK_PATH`: The path for storing DEGSS PPI network data.
- `ANNOT_PATH`: The path to the samples annotation CSV file.
- `SAMPLE_CONTRAST_FACTOR`: The factor used to contrast different sample types.
- `GOI_LEVEL_PREFIX`: The prefix used for the levels of the gene of interest.
- `CONTRAST_COMPARISONS`: A dictionary mapping contrast levels to their respective filters.
- `PQ_PAIRS`: An iterable of (p, q) pairs used for clustering.
- `P_COLS`: An iterable of p-value columns used for analysis.
- `P_THS`: An iterable of p-value thresholds used for analysis.
- `LFC_LEVELS`: An iterable of log fold change levels used for analysis.
- `LFC_THS`: An iterable of log fold change thresholds used for analysis.
- `MSIGDB_CATS`: An iterable of MSigDB categories used for analysis.
- `INTERACTION_SCORES`: An iterable of interaction scores used for PPI network analysis.
- `NETWORK_TYPES`: An iterable of network types used for PPI network analysis.
- `PARALLEL`: A boolean indicating whether to run processes in parallel.

### `src/pipelines/functional_analysis/run/rna_seq/pcta_wcdt/goi_clustering_ora.py`:

This script runs ORA on many gene sets collections (e.g. MSigDB H and C1-C8, DO, GO, Reactome, KEGG, MKEGG, and NCG) for all genes in each PPI cluster, obtaining enriched gene sets in the process.

**Global Variables (user-provided):**
- `STORAGE`: The root directory for storing data.
- `GOI_ENSEMBL`: The Ensembl ID for the gene of interest (FOLH1/PSMA).
- `SPECIES`: The species name, set to "Homo sapiens".
- `GOI_SYMBOL`: The symbol of the gene of interest.
- `DATA_ROOT`: The root path for storing data related to a specific dataset and gene of interest.
- `DATA_PATH`: The path for storing data files.
- `DESEQ_PATH`: The path for storing DESeq2 results.
- `PPI_NETWORK_PATH`: The path for storing DEGSS PPI network data.
- `ANNOT_PATH`: The path to the samples annotation CSV file.
- `SAMPLE_CONTRAST_FACTOR`: The factor used to contrast different sample types.
- `GOI_LEVEL_PREFIX`: The prefix used for the levels of the gene of interest.
- `CONTRAST_COMPARISONS`: A dictionary mapping contrast levels to their respective filters.
- `PQ_PAIRS`: An iterable of (p, q) pairs used for clustering.
- `P_COLS`: An iterable of p-value columns used for analysis.
- `P_THS`: An iterable of p-value thresholds used for analysis.
- `LFC_LEVELS`: An iterable of log fold change levels used for analysis.
- `LFC_THS`: An iterable of log fold change thresholds used for analysis.
- `MSIGDB_CATS`: An iterable of MSigDB categories used for analysis.
- `INTERACTION_SCORES`: An iterable of interaction scores used for PPI network analysis.
- `NETWORK_TYPES`: An iterable of network types used for PPI network analysis.
- `PARALLEL`: A boolean indicating whether to run processes in parallel.

<p align="right">(<a href="#top">back to top</a>)</p>

## Additional Notes

Source files formatted using the following commands:

```bash
ruff check . --fix && ruff format .
```

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->

## Contact

[Carlos Uziel Pérez Malla](https://perez-malla.com/)

[GitHub](https://github.com/CarlosUziel) - [Google Scholar](https://scholar.google.co.uk/citations?user=tEz_OeIAAAAJ&hl) - [LinkedIn](https://www.linkedin.com/in/carlosuziel)

<p align="right">(<a href="#top">back to top</a>)</p>

## Acknowledgments

This work was supported by grants provided by the FWF (National Science Foundation in Austria, grant no. P 32771) and DOC Fellowship of the Austrian Academy of Sciences (25276). The support from my PhD supervisors, Dr. Raheleh Sheibani and Prof. Dr. Gerda Egger was invaluable.

<p align="right">(<a href="#top">back to top</a>)</p>

## Citation

If you have found the content of this repository useful, please consider citing [this work](hhttps://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-025-06109-0):

```raw
@article{perez2025goistrat,
  title={Goistrat: gene-of-interest-based sample stratification for the evaluation of functional differences},
  author={P{\'e}rez Malla, Carlos Uziel and Kalla, Jessica and Tiefenbacher, Andreas and Wasinger, Gabriel and Kluge, Kilian and Egger, Gerda and Sheibani-Tezerji, Raheleh},
  journal={BMC bioinformatics},
  volume={26},
  number={1},
  pages={97},
  year={2025},
  publisher={Springer}
}
```

<p align="right">(<a href="#top">back to top</a>)</p>
