import json
from collections import defaultdict
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import rpy2.robjects as ro
import seaborn as sns
from components.functional_analysis.orgdb import OrgDB
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from r_wrappers.deseq2 import filter_dds, get_deseq_dataset_htseq, vst_transform
from r_wrappers.msigdb import get_msigb_gene_sets, get_msigdbr
from r_wrappers.tcgabiolinks import (
    gdc_download,
    gdc_prepare,
    gdc_query,
    get_query_results,
)
from r_wrappers.utils import (
    assay_to_df,
    rpy2_df_to_pd_df,
    rpy2_df_to_pd_df_manual,
    save_rds,
)
from rpy2.robjects.packages import importr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm.rich import tqdm

from data.io import copy_file, intersect_raw_counts, rename_genes, subset_star_counts
from data.ml import get_gene_set_expression_data
from data.utils import gene_expression_levels, parallelize_star
from data.visualization import gene_expression_plot


def tcga_rna_seq(project_name: str, data_path: Path, counts_path: Path) -> None:
    """
    Download and process RNASeq data of the a TCGA dataset.

    Args:
        data_path: Root data directory to store a TCGA dataset.
        counts_path: Directory to store count files.
    """

    # 1. GDC Query
    query = gdc_query(
        project=project_name,
        data_category="Transcriptome Profiling",
        data_type="Gene Expression Quantification",
        workflow_type="STAR - Counts",
        sample_type=ro.StrVector(["Solid Tissue Normal", "Primary Tumor"]),
    )

    # 1.1. Save query results
    query_results = get_query_results(query)
    records_file = data_path.joinpath("query_results.csv")
    query_results.to_csv(records_file, index=False)

    # 1.2. Build sample annotation
    sample_type_map = {"Solid Tissue Normal": "norm", "Primary Tumor": "prim"}

    samples_annotation = pd.DataFrame()
    samples_annotation["sample_id"] = query_results["sample.submitter_id"]
    samples_annotation["patient_id"] = query_results["cases.submitter_id"]
    samples_annotation["sample_type"] = [
        sample_type_map.get(x, x) for x in query_results["sample_type"]
    ]
    samples_annotation["file_name"] = [
        f + ".tsv" for f in query_results["sample.submitter_id"]
    ]
    samples_annotation["release_year"] = [
        x.split("-")[0] for x in query_results["updated_datetime"]
    ]

    # 1.3. Remove duplicated rows and columns
    samples_annotation = samples_annotation.loc[
        ~samples_annotation.duplicated(keep="first"),
        ~samples_annotation.columns.duplicated(keep="first"),
    ]

    # 1.4. Save sample annotation to disk
    samples_annotation.to_csv(data_path.joinpath("samples_annotation.csv"), index=False)

    # 2. Download data
    gdc_download(query=query, directory=str(data_path))

    # 2.1. Copy and rename counts files
    file_map = dict(zip(query_results["file_name"], samples_annotation["file_name"]))
    _ = parallelize_star(
        copy_file,
        [
            (counts_file, counts_path.joinpath(file_map[counts_file.name]))
            for counts_file in data_path.glob("**/*star_gene_counts.tsv")
        ],
        method="fork",
    )

    # 2.2. Filter relevant rows and columns from STAR counts files
    _ = parallelize_star(
        subset_star_counts,
        list(product(counts_path.glob("TCGA*.tsv"), [3])),
        method="fork",
    )

    # 2.3. Remove decimal part from ENSEMBL gene names, if any
    _ = parallelize_star(
        rename_genes, [(path,) for path in counts_path.glob("*.tsv")], method="fork"
    )

    # 3. Create raw counts dataset from counts files
    counts_df = pd.concat(
        [
            pd.read_csv(f, sep="\t", index_col=0, names=[None, f.stem])
            for f in counts_path.glob("*.tsv")
        ],
        axis=1,
    )

    # 3.1. Remove duplicated rows and columns
    counts_df = counts_df.loc[
        ~counts_df.index.duplicated(keep="first"),
        ~counts_df.columns.duplicated(keep="first"),
    ]

    # 3.2. Save raw counts to disk
    counts_df.to_csv(data_path.joinpath("raw_counts.csv"))

    # 4. Prepare data and save to disk
    save_path = data_path.joinpath("data.RDS")
    data = gdc_prepare(
        query=query,
        save=True,
        save_filename=str(save_path),
        directory=str(data_path),
        summarizedExperiment=True,
    )

    # 4.1. Save clinical data to disk
    col_data = data.do_slot("colData")
    rpy2_df_to_pd_df_manual(col_data).set_index("barcode").to_csv(
        data_path.joinpath("clinical_data.csv")
    )


def tcga_prad_meth_array(data_path: Path, idat_path: Path) -> None:
    """
    Download and process methylation array data of the TCGA-PRAD dataset.

    Args:
        data_path: Root data directory to store TCGA-PRAD dataset.
        idat_path: Directory to store .idat files.
    """

    # 1. GDC Query
    query = gdc_query(
        project="TCGA-PRAD",
        data_category="Raw microarray data",
        data_type="Raw intensities",
        experimental_strategy="Methylation array",
        legacy=True,
        file_type=".idat",
        platform="Illumina Human Methylation 450",
    )

    # 1.1. Save query results
    query_results = get_query_results(query)
    records_file = data_path.joinpath("query_results.csv")
    query_results.to_csv(records_file, index=False)

    # 1.2. Build sample annotation
    sample_type_map = {"Solid Tissue Normal": "norm", "Primary Tumor": "prim"}

    samples_annotation = pd.DataFrame()
    samples_annotation["sample_id"] = query_results["sample.submitter_id"]
    samples_annotation["patient_id"] = query_results["cases.submitter_id"]
    samples_annotation["sample_type"] = [
        sample_type_map.get(x, x) for x in query_results["sample_type"]
    ]
    samples_annotation["slide"] = [x.split("_")[0] for x in query_results["file_name"]]
    samples_annotation["array"] = [x.split("_")[1] for x in query_results["file_name"]]
    samples_annotation["probe"] = [
        x.split("_")[2].split(".")[0] for x in query_results["file_name"]
    ]
    samples_annotation["barcode"] = [
        "_".join(x.split("_")[:2]) for x in query_results["file_name"]
    ]
    samples_annotation["file_name"] = query_results["file_name"]
    samples_annotation["release_year"] = [
        x.split("-")[0] for x in query_results["updated_datetime"]
    ]
    samples_annotation_file = data_path.joinpath("samples_annotation.csv")
    samples_annotation.to_csv(samples_annotation_file, index=False)

    # 1.3. Keep the same samples for which we have RNASeq data
    query_rna = gdc_query(
        project="TCGA-PRAD",
        data_category="Transcriptome Profiling",
        data_type="Gene Expression Quantification",
        workflow_type="STAR - Counts",
        sample_type=ro.StrVector(["Solid Tissue Normal", "Primary Tumor"]),
    )
    rnaseq_query_results = get_query_results(query_rna)

    samples_annotation_filtered = samples_annotation[
        samples_annotation["sample_id"].isin(
            rnaseq_query_results["sample.submitter_id"]
        )
    ]
    save_path = data_path.joinpath("samples_annotation_common.csv")
    samples_annotation_filtered.to_csv(save_path, index=False)

    # 2. Download data
    gdc_download(query=query, directory=str(data_path))

    # 2.1. Copy .idat files to same directory
    _ = parallelize_star(
        copy_file,
        [
            (idat_file, idat_path.joinpath(idat_file.name))
            for idat_file in data_path.joinpath("TCGA-PRAD").glob("**/*.idat")
        ],
        method="fork",
    )

    # 3. Prepare data and save to disk
    r_sesame_data = importr("sesameData")
    r_sesame_data.sesameDataCache("EPIC")
    r_sesame_data.sesameDataCache("HM450")

    save_path = data_path.joinpath("data.RDS")
    data = gdc_prepare(
        query=query,
        save=True,
        save_filename=str(save_path),
        directory=str(data_path),
        summarizedExperiment=True,
    )

    # 3.1. Save clinical data to disk
    col_data = data.do_slot("colData")
    save_path = data_path.joinpath("clinical_data.csv")
    rpy2_df_to_pd_df_manual(col_data).set_index("barcode").to_csv(save_path)


def su2c_pcf_annotation(
    data_path: Path,
    rna_fastq_path: Optional[Path] = None,
    dna_fastq_path: Optional[Path] = None,
) -> None:
    """
    Gather and unify all the different annotations for the samples of the
    SU2C-PCF dataset (dbGAP accession number: phs000915.v2.p2).

    Original annotation files must first be downloaded and placed (uncompressed)
    in `data_path`:
        dbGAP basic annotation (stored along the sample files).
        SRA metadata:
            https://www.ncbi.nlm.nih.gov/Traces/study/?acc=phs000915
        Clinical metadata:
            https://cbioportal-datahub.s3.amazonaws.com/prad_su2c_2019.tar.gz)

    Annotations for all data types, RNASeq and WES, are processed.

    Args:
        data_path: Root data directory containing all relevant annotation files.
        rna_fastq_path: Path to SU2C-PCF RNA FASTQ files.
        dna_fastq_path: Path to SU2C-PCF DNA FASTQ files.
    """

    # 1. Setup
    # Prostate Cancer Specific (PCS)
    pcs_path = data_path.joinpath(Path("70655_PCS"))

    # Cancer Specific (CSD)
    csd_path = data_path.joinpath(Path("71100_CSD"))

    # General Research (GRU)
    gru_path = data_path.joinpath(Path("71102_GRU"))

    # 1.1. Clinical annotations
    df1 = pd.read_csv(
        data_path.joinpath("prad_su2c_2019/data_clinical_patient.txt"),
        sep="\t",
        comment="#",
        index_col="PATIENT_ID",
    )
    df2 = pd.read_csv(
        data_path.joinpath("prad_su2c_2019/data_clinical_sample.txt"),
        sep="\t",
        comment="#",
        index_col="PATIENT_ID",
    )  # this one has more than 1 sample for some patients
    clinical_annot_df = df1.join(df2).reset_index()
    clinical_annot_df = clinical_annot_df[
        ~clinical_annot_df["PATIENT_ID"].duplicated(keep=False)
    ]

    # 1.2. SRA Metadata
    sra_metadata_df = pd.read_csv(
        data_path.joinpath("SraRunTable.txt"), sep=",", comment="#"
    ).dropna(axis=1)

    sra_metadata_clinical_df = pd.merge(
        sra_metadata_df,
        clinical_annot_df.reset_index(),
        left_on="submitted_subject_id",
        right_on="PATIENT_ID",
        how="left",
    )
    sra_metadata_clinical_df.set_index("Run").to_csv(
        data_path.joinpath("sra_metadata_clinical.csv")
    )
    sra_metadata_clinical_df.replace("UNK", np.nan, inplace=True)

    # 2. Process each patient group separately
    # 2.1. Convert .txt annotation files into pandas dataframes
    patient_groups_dfs = {
        patient_path.name: [
            pd.read_csv(file_txt, sep="\t", comment="#")
            for file_txt in patient_path.glob("**/*.txt")
        ]
        for patient_path in (pcs_path, csd_path, gru_path)
    }

    # 2.2. Merge all dataframes per sample group
    merged_dfs = {}
    for patient_group, dfs in patient_groups_dfs.items():
        merged_df = dfs[0]
        for df in dfs[1:]:
            merged_df = merged_df.merge(
                df, on=merged_df.columns.intersection(df.columns).to_list()
            )
        merged_dfs[patient_group] = pd.merge(
            sra_metadata_clinical_df,
            merged_df,
            left_on="biospecimen_repository_sample_id",
            right_on="SAMPLE_ID",
        )

    # 2.3. Save merged dataframes
    for patient_group, merged_df in merged_dfs.items():
        merged_df.set_index("Run").to_csv(
            data_path.joinpath(f"{patient_group}_annotation.csv")
        )

    # 3. All patient groups
    all_df = pd.concat(merged_dfs.values()).set_index("Run")
    # rename endocrime tumours accordingly
    all_df.loc[all_df["NEUROENDOCRINE_FEATURES"] == "Yes", "SAMPLE_TYPE"] = "endo"
    all_df.to_csv(data_path.joinpath("samples_annotation.csv"))

    # 3.1. RNA samples
    all_df_rna = all_df[all_df["ANALYTE_TYPE"].str.contains("RNA")]
    all_df_rna.to_csv(data_path.joinpath("samples_annotation_rna.csv"))

    # 3.1.1. Filter by downloadable samples
    if rna_fastq_path:
        downloaded_run_ids = set([p.stem for p in rna_fastq_path.glob("**")])
        all_df_rna_available = all_df_rna.loc[
            all_df_rna.index.intersection(downloaded_run_ids), :
        ]

        all_df_rna_available
        all_df_rna_available.to_csv(
            data_path.joinpath("samples_annotation_rna_downloaded.csv")
        )

    # 3.2. DNA samples
    all_df_dna = all_df[all_df["ANALYTE_TYPE"].str.contains("DNA")]
    all_df_dna.to_csv(data_path.joinpath("samples_annotation_dna.csv"))

    # 3.2.1. Filter by downloadable samples
    if dna_fastq_path:
        downloaded_run_ids = set([p.stem for p in dna_fastq_path.glob("**")])
        all_df_dna_available = all_df_dna.loc[
            all_df_dna.index.intersection(downloaded_run_ids), :
        ]
        all_df_dna_available
        all_df_dna_available.to_csv(
            data_path.joinpath("samples_annotation_dna_downloaded.csv")
        )

    # 4. Standarized samples annotation for differential expression
    samples_annotation_su2c = pd.DataFrame()
    samples_annotation_su2c["sample_id"] = all_df_rna_available.index
    samples_annotation_su2c["patient_id"] = list(all_df_rna_available["SUBJECT_ID"])
    samples_annotation_su2c["sample_type"] = [
        "met" if sample_type == "Tumor" else sample_type
        for sample_type in all_df_rna_available["SAMPLE_TYPE"]
    ]
    samples_annotation_su2c["file_name"] = [
        f + ".tsv" for f in all_df_rna_available.index
    ]
    samples_annotation_su2c["release_year"] = [
        x.split("-")[0] for x in all_df_rna_available["ReleaseDate"]
    ]
    samples_annotation_su2c.set_index("sample_id").to_csv(
        data_path.joinpath("samples_annotation_rna_downloaded_standarized.csv")
    )


def tcga_prad_su2c_pcf_rna_seq(
    data_path: Path,
    deseq2_path: Path,
    plots_path: Path,
    tcga_prad_annot_file: Path,
    su2c_pcf_annot_file: Path,
    sample_colors: Dict[str, str],
    sample_type_col: str = "sample_type",
) -> None:
    """
    Combine and process RNASeq data from TCGA-PRAD and SU2C-PCF (dbGAP accesion
        number: phs000915.v2.p2) datasets.

    Args:
        data_path: Root data directory containing TCGA-PRAD and SU2C-PCF datasets.
        deseq2_path: Directory to store preliminary DESeq2 results.
        plots_path: Directory to store intermediate plots.
        tcga_prad_annot_file: TCGA-PRAD annotation file.
        su2c_pcf_annot_file: SU2C-PCF annotation file.
        sample_colors. Color palette used for plotting.
        sample_type_col: ID column name used to differentiate sample types.
    """

    # 0. Setup
    counts_path = data_path.joinpath("star_counts")
    plots_path.mkdir(exist_ok=True, parents=True)
    tcga_prad_annot_df = pd.read_csv(tcga_prad_annot_file)
    su2c_pcf_annot_df = pd.read_csv(su2c_pcf_annot_file)

    # 1. Process data files
    # 1.1. Combine annotation files
    samples_annotation_all = pd.concat(
        [tcga_prad_annot_df, su2c_pcf_annot_df]
    ).set_index("sample_id")

    samples_annotation_all.to_csv(
        data_path.joinpath("samples_annotation_tcga_prad_su2c.csv")
    )

    # 1.2. Remove decimal part from ENSEMBL gene names, if any
    _ = parallelize_star(
        rename_genes, [(path,) for path in counts_path.glob("*.tsv")], method="fork"
    )

    # 1.3. Remove genes not present in all files
    intersect_raw_counts(counts_path)

    # 2. Generate DESeq2 data
    # 2.1. Annotation data
    annot_df = deepcopy(samples_annotation_all)

    # 2.2. DESeq2 dataset
    dds = filter_dds(
        get_deseq_dataset_htseq(
            annot_df=annot_df,
            counts_path=counts_path,
            factors=[sample_type_col],
            design_factors=["1"],  # special design factor for no DE
            counts_files_pattern="*.tsv",
        )
    )
    dds_df = rpy2_df_to_pd_df(ro.r("counts")(dds))

    # 2.2.1. Save to disk
    sample_types = sorted(set(annot_df[sample_type_col]))
    sample_types_str = "+".join(sample_types)

    save_path = deseq2_path.joinpath(f"{sample_type_col}_{sample_types_str}_dds.RDS")
    save_rds(dds, save_path)
    save_path = deseq2_path.joinpath(f"{sample_type_col}_{sample_types_str}_dds.csv")
    dds_df.to_csv(save_path)

    # 2.3. Variance Stabilizing Transform (VST)
    vst = vst_transform(dds)
    vst_df = assay_to_df(vst)

    # 2.3.1. Save to disk
    save_path = deseq2_path.joinpath(f"{sample_type_col}_{sample_types_str}_vst.RDS")
    save_rds(vst, save_path)
    save_path = deseq2_path.joinpath(f"{sample_type_col}_{sample_types_str}_vst.csv")
    vst_df.to_csv(save_path)

    # 2.4. PCA plot of all samples
    pca = PCA(n_components=2, random_state=8080)
    components = pca.fit_transform(vst_df.transpose())
    ratios = pca.explained_variance_ratio_ * 100

    labels = annot_df.loc[vst_df.columns, "sample_type"]
    fig = px.scatter(
        components,
        x=0,
        y=1,
        labels={"0": f"PC 1 ({ratios[0]:.2f}%)", "1": f"PC 2 ({ratios[1]:.2f}%)"},
        color=labels,
        color_discrete_map=sample_colors,
        hover_name=vst_df.columns,
        title="All Samples (VST)",
    )
    fig.write_image(plots_path.joinpath("global_pca_vst.pdf"))
    fig.write_html(plots_path.joinpath("global_pca_vst.html"))
    plt.clf()

    # 2.4. PCA plot of each sample type
    for sample_type in sample_types:
        vst_pca_df = vst_df.loc[
            :, annot_df[annot_df[sample_type_col] == sample_type].index
        ]
        pca = PCA(n_components=2, random_state=8080)
        components = pca.fit_transform(vst_pca_df.transpose())
        ratios = pca.explained_variance_ratio_ * 100

        labels = annot_df.loc[vst_pca_df.columns, sample_type_col]
        fig = px.scatter(
            components,
            x=0,
            y=1,
            labels={"0": f"PC 1 ({ratios[0]:.2f}%)", "1": f"PC 2 ({ratios[1]:.2f}%)"},
            color=labels,
            color_discrete_map=sample_colors,
            hover_name=vst_pca_df.columns,
            title="All Samples (VST)",
        )
        fig.write_image(plots_path.joinpath(f"{sample_type}_pca_vst.pdf"))
        fig.write_html(plots_path.joinpath(f"{sample_type}_pca_vst.html"))
        plt.clf()


def su2c_pcf_clusters(
    root_path: Path,
    annot_file: Path,
    sample_colors: Dict[str, str],
    sample_type_col: str,
    sample_cluster_field: str,
) -> None:
    """
    Identify and annoate SU2C-PCF clusters.

    Args:
        root_path: Project root directory.
        annot_file: Annotation file.
        sample_colors. Color palette used for plotting.
        sample_type_col: ID column name used to differentiate sample types.
        sample_cluster_field: Column name of final clustering results.
    """
    # 0. Setup
    data_path = root_path.joinpath("data")
    counts_path = data_path.joinpath("star_counts")
    deseq2_path = root_path.joinpath("deseq2")
    plots_path = root_path.joinpath("plots")
    annot_df = pd.read_csv(annot_file, index_col=0)
    sample_types = sorted(set(annot_df[sample_type_col]))
    sample_types_str = "+".join(sample_types)

    vst_df = pd.read_csv(
        deseq2_path.joinpath(f"{sample_type_col}_{sample_types_str}_vst.csv"),
        index_col=0,
    )

    # 1. Identify clusters
    vst_met_df = vst_df.loc[:, annot_df[annot_df[sample_type_col] == "met"].index]
    pca = PCA(n_components=2, random_state=8080)
    components = pca.fit_transform(vst_met_df.transpose())
    ratios = pca.explained_variance_ratio_ * 100

    kmeans = KMeans(n_clusters=2, random_state=8080).fit(components)
    labels = ["AB"[x] for x in kmeans.labels_]

    fig = px.scatter(
        components,
        x=0,
        y=1,
        labels={"0": f"PC 1 ({ratios[0]:.2f}%)", "1": f"PC 2 ({ratios[1]:.2f}%)"},
        color=labels,
        hover_name=vst_met_df.columns,
        title="met samples (VST)",
    )
    fig.write_image(plots_path.joinpath("met_clusters_pca_vst.pdf"))
    fig.write_html(plots_path.joinpath("met_clusters_pca_vst.html"))
    plt.clf()

    # 1.1. Annotate identified clusters
    annot_df["sample_cluster"] = annot_df["sample_type"]
    annot_df.loc[vst_met_df.columns, "sample_cluster"] = [
        f"met_{label}" for label in labels
    ]

    # 1.2. Correlation heatmap of clusters
    annot_df_heatmap = deepcopy(
        annot_df[annot_df["sample_cluster"].isin(("met_a", "met_b"))].sort_values(
            "sample_cluster"
        )
    )

    vst_heatmap = vst_met_df.loc[:, annot_df_heatmap.index].corr()
    vst_heatmap = vst_heatmap.sub(vst_heatmap.mean(axis=1), axis=0)
    corr_matrix = vst_heatmap.corr()

    cluster_colors = annot_df_heatmap["sample_cluster"].map(sample_colors).rename("")

    fig = plt.figure(figsize=(10, 10), dpi=300)
    g = sns.clustermap(
        corr_matrix,
        center=0,
        cmap="vlag",
        row_colors=cluster_colors,
        col_colors=cluster_colors,
        cbar_pos=(0.1, 0.35, 0.03, 0.2),
        vmin=-1,
        vmax=1,
        yticklabels=False,
        xticklabels=False,
    )

    g.ax_row_dendrogram.remove()
    g.ax_col_dendrogram.remove()

    handles = [
        Patch(facecolor=sample_colors[name])
        for name in set(annot_df_heatmap["sample_cluster"])
    ]
    plt.legend(
        handles,
        {k: sample_colors[k] for k in set(annot_df_heatmap["sample_cluster"])},
        title="sample_cluster",
        bbox_transform=plt.gcf().transFigure,
        bbox_to_anchor=(1.15, 0.5),
    )
    plt.savefig(
        plots_path.joinpath("met_clusters_heatmap_vst.pdf"), bbox_inches="tight"
    )

    # 2. Remove replicates from clusters
    met_A_no_replicates = (
        annot_df[annot_df["sample_cluster"] == "met_a"]
        .sort_values("release_year", ascending=False)
        .drop_duplicates("patient_id")
        .index
    )
    met_B_no_replicates = (
        annot_df[annot_df["sample_cluster"] == "met_b"]
        .sort_values("release_year", ascending=False)
        .drop_duplicates("patient_id")
        .index
    )

    annot_df[sample_cluster_field] = annot_df["sample_cluster"]
    annot_df.loc[met_A_no_replicates, sample_cluster_field] = "met_aa"
    annot_df.loc[met_B_no_replicates, sample_cluster_field] = "met_bb"

    vst_met_filtered = vst_met_df.loc[
        :,
        annot_df[annot_df[sample_cluster_field].isin(("met_aa", "met_bb"))].index,
    ]

    # 2.2. PCA of unique patients
    pca = PCA(n_components=2, random_state=8080)
    components = pca.fit_transform(vst_met_filtered.transpose())
    ratios = pca.explained_variance_ratio_ * 100

    labels = annot_df.loc[vst_met_filtered.columns, sample_cluster_field]
    fig = px.scatter(
        components,
        x=0,
        y=1,
        labels={"0": f"PC 1 ({ratios[0]:.2f}%)", "1": f"PC 2 ({ratios[1]:.2f}%)"},
        color=labels,
        color_discrete_map=sample_colors,
        hover_name=vst_met_filtered.columns,
        title="met Samples without replicates (VST)",
    )
    fig.write_image(plots_path.joinpath("met_clusters_no_replicates_pca_vst.pdf"))
    fig.write_html(plots_path.joinpath("met_clusters_no_replicates_pca_vst.html"))
    plt.clf()

    # 2.3. Correlation heatmap of clusters without replicates
    annot_df_heatmap = deepcopy(
        annot_df[annot_df[sample_cluster_field].str.contains("met")].sort_values(
            sample_cluster_field
        )
    )

    vst_heatmap = vst_met_df.loc[:, annot_df_heatmap.index].corr()
    vst_heatmap = vst_heatmap.sub(vst_heatmap.mean(axis=1), axis=0)
    corr_matrix = vst_heatmap.corr()

    cluster_colors = (
        annot_df_heatmap[sample_cluster_field].map(sample_colors).rename("")
    )

    fig = plt.figure(figsize=(10, 10), dpi=300)
    g = sns.clustermap(
        corr_matrix,
        center=0,
        cmap="vlag",
        row_colors=cluster_colors,
        col_colors=cluster_colors,
        cbar_pos=(0.1, 0.35, 0.03, 0.2),
        vmin=-1,
        vmax=1,
        yticklabels=False,
        xticklabels=False,
    )

    g.ax_row_dendrogram.remove()
    g.ax_col_dendrogram.remove()

    handles = [
        Patch(facecolor=sample_colors[name])
        for name in set(annot_df_heatmap[sample_cluster_field])
    ]
    plt.legend(
        handles,
        {k: sample_colors[k] for k in set(annot_df_heatmap[sample_cluster_field])},
        title=sample_cluster_field,
        bbox_transform=plt.gcf().transFigure,
        bbox_to_anchor=(1.25, 0.5),
    )
    plt.savefig(
        plots_path.joinpath("met_clusters_no_replicates_heatmap_vst.pdf"),
        bbox_inches="tight",
    )

    # 3. Recompute DESeq2 datasets for each sample cluster
    # 3.1. DESeq2 dataset
    dds = filter_dds(
        get_deseq_dataset_htseq(
            annot_df=annot_df,
            counts_path=counts_path,
            factors=[sample_cluster_field],
            design_factors=["1"],  # special design factor for no DE
            counts_files_pattern="*.tsv",
        )
    )
    dds_df = rpy2_df_to_pd_df(ro.r("counts")(dds))

    # 3.1.1. Save to disk
    sample_clusters = sorted(set(annot_df[sample_cluster_field]))
    sample_clusters_str = "+".join(sample_clusters)

    save_path = deseq2_path.joinpath(
        f"{sample_cluster_field}_{sample_clusters_str}_dds.RDS"
    )
    save_rds(dds, save_path)
    save_path = deseq2_path.joinpath(
        f"{sample_cluster_field}_{sample_clusters_str}_dds.csv"
    )
    dds_df.to_csv(save_path)

    # 3.2. Variance Stabilizing Transform (VST)
    vst = vst_transform(dds)
    vst_df = assay_to_df(vst)

    # 3.2.1. Save to disk
    save_path = deseq2_path.joinpath(
        f"{sample_cluster_field}_{sample_clusters_str}_vst.RDS"
    )
    save_rds(vst, save_path)
    save_path = deseq2_path.joinpath(
        f"{sample_cluster_field}_{sample_clusters_str}_vst.csv"
    )
    vst_df.to_csv(save_path)

    # 3.3. PCA of all samples
    pca = PCA(n_components=2, random_state=8080)
    components = pca.fit_transform(vst_df.transpose())
    ratios = pca.explained_variance_ratio_ * 100

    labels = annot_df.loc[vst_df.columns, sample_cluster_field]
    fig = px.scatter(
        components,
        x=0,
        y=1,
        labels={"0": f"PC 1 ({ratios[0]:.2f}%)", "1": f"PC 2 ({ratios[1]:.2f}%)"},
        color=labels,
        color_discrete_map=sample_colors,
        hover_name=vst_df.columns,
        title="All samples (no replicates) (VST)",
    )
    fig.write_image(plots_path.joinpath("pca_clusters_vst.pdf"))
    fig.write_html(plots_path.joinpath("pca_clusters_vst.html"))
    plt.clf()

    # 4. Save final annotation file
    annot_df.to_csv(
        data_path.joinpath("samples_annotation_tcga_prad_su2c_clusters.csv")
    )


def goi_perc_annotation_rna_seq(
    annot_df: pd.DataFrame,
    vst_df: pd.DataFrame,
    plots_path: Path,
    data_path: Path,
    new_annot_file: Path,
    goi_symbol: str,
    sample_contrast_factor: str,
    contrast_levels: Iterable[str],
    contrast_levels_colors: Dict[str, str],
    percentiles: Tuple = (10, 20, 30),
    pca_top_n: int = 1000,
) -> None:
    """
    Gene of interest (GOI) percentile annotation (per sample type) of RNASeq samples.

    Args:
        annot_df: Dataframe of sample annotations.
        vst_df: Dataframe of VST transformed gene counts.
        plots_path: Directory to store intermediate plots.
        data_path: Root data directory.
        new_annot_file: New annotation file with GOI expression levels.
        goi_ensembl: GOI gene ENSEMBL ID.
        goi_symbol: GOI gene SYMBOL ID.
        sample_contrast_factor: Annotated field used for group splits (contrast).
        contrast_levels: Contrast levels to be split.
        contrast_levels_colors: Color palette used for plotting.
        percentiles: Percentiles used to divide ranked list of samples. A value of 10
            means that 10% of the bottom and top samples (ranked by GOI expression) will
            be assigned to low and high groups, respectively. The rest will be assigned
            to the mid group.
        pca_top_n: How many features to use when calculating PCA before plotting.
    """
    # 1. Use GOI counts to determine expression levels (i.e., high, mid and low)
    for contrast_level, percentile in product(contrast_levels, percentiles):
        levels_col = f"{goi_symbol}_level_{percentile}"
        annot_df_with_levels = gene_expression_levels(
            expr_df=annot_df[annot_df[sample_contrast_factor] == contrast_level],
            gene_expr_col=f"{goi_symbol}_VST",
            gene_expr_level=levels_col,
            percentile=percentile,
        )
        annot_df.loc[annot_df[sample_contrast_factor] == contrast_level, levels_col] = (
            annot_df_with_levels[levels_col]
        )

    # 1.1. Plot GOI VST counts
    for contrast_level, percentile in product(contrast_levels, percentiles):
        levels_col = f"{goi_symbol}_level_{percentile}"
        title = f"{goi_symbol} - {contrast_level} Samples"

        save_path = plots_path.joinpath(
            f"{contrast_level}_{levels_col}_expression_plot.pdf"
        )
        _ = gene_expression_plot(
            annot_df[annot_df[sample_contrast_factor] == contrast_level],
            save_path,
            title,
            gene_expr_col=f"{goi_symbol}_VST",
            gene_expr_level=levels_col,
        )

        save_path = plots_path.joinpath(
            f"{contrast_level}_{levels_col}_expression_plot.html"
        )
        _ = gene_expression_plot(
            annot_df[annot_df[sample_contrast_factor] == contrast_level],
            save_path,
            title,
            gene_expr_col=f"{goi_symbol}_VST",
            gene_expr_level=levels_col,
        )

    # 1.2. PCA plot of each sample cluster (top most variable genes)
    for contrast_level in contrast_levels:
        vst_pca_df = vst_df.loc[
            :, annot_df[annot_df[sample_contrast_factor] == contrast_level].index
        ]
        pca = PCA(n_components=2, random_state=8080)
        top_genes = vst_pca_df.std(axis=1).sort_values(ascending=False)[:pca_top_n]
        components = pca.fit_transform(vst_pca_df.loc[top_genes.index].transpose())
        ratios = pca.explained_variance_ratio_ * 100

        for percentile in percentiles:
            levels_col = f"{goi_symbol}_level_{percentile}"
            labels = annot_df.loc[vst_pca_df.columns, levels_col]

            fig = px.scatter(
                components,
                x=0,
                y=1,
                labels={
                    "0": f"PC 1 ({ratios[0]:.2f}%)",
                    "1": f"PC 2 ({ratios[1]:.2f}%)",
                },
                color=labels,
                color_discrete_map=contrast_levels_colors,
                hover_name=vst_pca_df.columns,
                title=(
                    f"{goi_symbol} (p={percentile}) expression levels for"
                    f" {contrast_level} (VST)"
                ),
            )
            fig.write_image(
                plots_path.joinpath(
                    f"{contrast_level}_{levels_col}_top_{pca_top_n}_pca_vst.pdf"
                )
            )
            fig.write_html(
                plots_path.joinpath(
                    f"{contrast_level}_{levels_col}_top_{pca_top_n}_pca_vst.html"
                )
            )
            plt.clf()

    # 2. GOI expression distribution among sample types
    plt.figure(facecolor="white", figsize=(8, 8), dpi=200)
    plt.xticks(fontsize=8)
    sns.boxplot(
        data=annot_df,
        x=sample_contrast_factor,
        y=f"{goi_symbol}_VST",
        notch=1,
    )
    sns.stripplot(data=annot_df, x=sample_contrast_factor, y=f"{goi_symbol}_VST")
    fig.write_image(
        plots_path.joinpath(f"{sample_contrast_factor}_{goi_symbol}_distribution.pdf")
    )
    fig.write_html(
        plots_path.joinpath(f"{sample_contrast_factor}_{goi_symbol}_distribution.html")
    )

    # 3. Save GOI annotation file
    annot_df.sort_values(sample_contrast_factor, ascending=False).to_csv(
        data_path.joinpath(new_annot_file)
    )


def goi_annotation_meth_array(
    annot_file_rna: Path,
    anno_file_meth: Path,
    idat_path: Path,
    data_path: Path,
    goi_symbol: str,
    percentiles: Tuple = (10, 20, 30),
) -> None:
    """
    GOI annotation (per sample type) of methylation array samples using percentile
    strategy.

    Args:
        annot_file_rna: Annotation file of RNA (RNA-Seq) samples.
        anno_file_meth: Annotation file of DNA Methylation samples.
        idat_path: Directory containing .idat files.
        data_path: Root data directory.
        goi_symbol: GOI gene SYMBOL ID.
        percentiles: Percentiles used to divide ranked list of samples. A 0.1 value
            means that 10% of the bottom and top samples (ranked by GOI expression) will
            be assigned to low and high groups, respectively. The rest will be assigned
            to the mid group.
    """
    # 0. Setup
    annot_df_rna = pd.read_csv(annot_file_rna, dtype=str)
    annot_df_meth = pd.read_csv(anno_file_meth, dtype=str)
    sample_types = set(annot_df_meth["sample_type"])

    # 1. Find matching entires
    # 1.1. Prepare RNA annotation
    # set multi-index (to be sure that sample ids match sample types)
    annot_df_rna = annot_df_rna[
        annot_df_rna["sample_type"].isin(sample_types)
    ].set_index(["sample_id", "sample_type"])

    # 1.2. Prepare MethArray annotation
    # keep only one entry per probe
    annot_df_meth = annot_df_meth.sort_values("probe").drop_duplicates("sample_id")

    # add basename field
    annot_df_meth["Basename"] = [
        str(idat_path.joinpath(barcode)) for barcode in annot_df_meth["barcode"]
    ]

    # set multi-index (to be sure that sample ids match sample types)
    annot_df_meth = annot_df_meth[
        annot_df_meth["sample_type"].isin(sample_types)
    ].set_index(["sample_id", "sample_type"])

    # 1.3. Match indices and assign gene levels
    indices = annot_df_rna.index.intersection(annot_df_meth.index)

    annot_df_meth.loc[indices, [f"{goi_symbol}_level_{p}" for p in percentiles]] = (
        annot_df_rna.loc[indices, [f"{goi_symbol}_level_{p}" for p in percentiles]]
    )

    # 2. Save results
    annot_df_meth.to_csv(
        data_path.joinpath(f"samples_annotation_common_{goi_symbol}.csv")
    )


def generate_gsva_matrix(
    counts_df: pd.DataFrame,
    annot_df: pd.DataFrame,
    org_db: OrgDB,
    msigdb_cat: str,
    save_path: str,
    contrast_factor: Optional[str] = None,
    gsva_threads: int = 8,
) -> None:
    """Given a raw counts gene matrix, generate its corresponding GSVA matrix for a
    given MSigDB category.

    Additionally, save the gene elements of each gene set.

    Args:
        counts_df: Counts gene matrix.
        annot_df: Samples annotation dataframe.
        org_db: Organism annotation database.
        msigdb_cat: Category of MSigDB to extract the gene sets from.
        save_path: .csv file path to save results GSVA matrix to.
        contrast_factor: Column name determining sample groups.
        gsva_threads: Number of threads to use while running GSVA. Defaults to 8.
    """

    assert save_path.suffix == ".csv", "Save path is not a .csv file"

    # 1. Get GSVA matrix
    gsva_matrix = get_gene_set_expression_data(
        counts=counts_df,
        annot_df=annot_df,
        contrast_factor=contrast_factor,
        org_db=org_db,
        msigdb_cat=msigdb_cat,
        gsva_threads=gsva_threads,
    )[0].transpose()

    gsva_matrix.to_csv(save_path)

    # 2. Get gene sets metadata such as gene members and description
    msigdb_genes = {}
    for gene_id in ("entrez_gene", "gene_symbol"):
        gene_sets_dict = get_msigb_gene_sets(org_db.species, msigdb_cat, gene_id)
        msigdb_genes[gene_id] = {
            gene_set: "/".join(map(str, genes))
            for gene_set, genes in gene_sets_dict.items()
        }
    msigdb_meta = pd.DataFrame(msigdb_genes)

    msigdb_meta["gs_description"] = (
        get_msigdbr(org_db.species, category=msigdb_cat)
        .drop_duplicates(subset=["gs_name"])
        .set_index("gs_name")
        .loc[msigdb_meta.index, "gs_description"]
    )
    msigdb_meta[["gs_description", "entrez_gene", "gene_symbol"]].to_csv(
        save_path.parent.joinpath(f"{msigdb_cat}_meta.csv")
    )


def get_optimal_gsva_splits(
    results_path: Path,
    msigdb_cats_meta_paths: Iterable[Path],
    group_counts: Iterable[Tuple[str, int, int, int]],
    goi_level_prefix: str,
    msigdb_cats: Iterable[str],
    sample_contrast_factor: str,
    contrasts_levels: Iterable[str],
    annot_df: pd.DataFrame,
    goi_symbol: str,
    annot_path_new: Path,
    p_col: str = "padj",
    p_th: float = 0.05,
    lfc_level: str = "all",
    lfc_th: float = 0.0,
) -> None:
    """
    Find the GSVA-based samples split that maximizes the functional difference score
        between low and high groups.

    Args:
        results_path: Directory to store final results to.
        msigdb_cats_meta_paths: Collection of paths to each MSigDB category metadata
            file.
        group_counts: Low, mid and high group counts for each sliding window iteration.
        goi_level_prefix: Annotation field name to identify GOI splits.
        msigdb_cats: MSigDB categories to be considered when computing the functional
            difference score.
        sample_contrast_factor: Annotated field used for group splittings (contrast).
        contrasts_levels: Contrast levels to be split.
        annot_df: Samples annotation dataframe.
        goi_symbol: GOI gene SYMBOL ID.
        annot_path_new: New annotation file to be generated.
        p_col: P-value column name to be used.
        p_th: P-value threshold.
        lfc_level: Log2 Fold Chance level (up, down or all de-regulated genes).
        lfc_th: Log2 Fold Chance threshold.
    """
    # 0. Setup
    sample_groups_summary = defaultdict(lambda: defaultdict(dict))
    p_th_str = str(p_th).replace(".", "_")
    lfc_th_str = str(lfc_th).replace(".", "_")
    msigdb_cats_meta_dfs = {
        msigdb_cat: pd.read_csv(msigdb_cats_meta_path, index_col=0)
        for msigdb_cat, msigdb_cats_meta_path in msigdb_cats_meta_paths.items()
    }

    # 1. Summarize results per sample group and MSigDB category
    sample_groups_summary = defaultdict(lambda: defaultdict(dict))
    for (contrast_level, low_n, mid_n, high_n), msigdb_cat in tqdm(
        list(product(group_counts, msigdb_cats))
    ):
        exp_prefix = f"{contrast_level}_{goi_level_prefix}_high_{high_n}+low_{low_n}_"

        # 1.1. Load D-GSVA results
        all_results_df = pd.read_csv(
            results_path.joinpath(msigdb_cat).joinpath(
                f"{exp_prefix}_high_vs_low_top_table_"
                f"{p_col}_{p_th_str}_{lfc_level}_{lfc_th_str}.csv"
            ),
            index_col=0,
        )

        # 1.2. Compute functional difference score
        sample_groups_summary[contrast_level][msigdb_cat][
            f"{low_n}_{mid_n}_{high_n}"
        ] = (len(all_results_df) / len(msigdb_cats_meta_dfs[msigdb_cat])) * np.sqrt(
            np.mean(np.power(all_results_df["log2FoldChange"], 2))
        )

    with results_path.joinpath("sample_groups_summary.json").open("w") as fp:
        json.dump(sample_groups_summary, fp, indent=4)

    # 2. Get best splits per sample type
    best_splits = {}
    annot_df_splits = []
    for contrast_level in contrasts_levels:
        # 2.1. Compute median functional score over all MSigDB collections
        split_stats_df = pd.DataFrame(sample_groups_summary[contrast_level]).fillna(0)
        split_stats_df.to_csv(results_path.joinpath(f"{contrast_level}_summary.csv"))
        best_split_str = (
            split_stats_df.median(axis=1).sort_values(ascending=False).index[0]
        )

        best_splits[contrast_level] = {
            k: v
            for k, v in zip(("low_n", "mid_n", "high_n"), best_split_str.split("_"))
        }

        # 2.2. Add group labels to annotation dataframe
        annot_df_sorted = deepcopy(
            annot_df[annot_df[sample_contrast_factor] == contrast_level].sort_values(
                f"{goi_symbol}_VST", ascending=True
            )
        )
        annot_df_sorted[goi_level_prefix] = (
            ["low"] * int(best_splits[contrast_level]["low_n"])
            + ["mid"] * int(best_splits[contrast_level]["mid_n"])
            + ["high"] * int(best_splits[contrast_level]["high_n"])
        )
        annot_df_splits.append(annot_df_sorted)

    # 3. Save best splits
    with results_path.joinpath("best_splits.json").open("w") as fp:
        json.dump(best_splits, fp, indent=4)

    # 4. Save annotation dataframe including best split per sample type
    pd.concat(annot_df_splits).to_csv(annot_path_new)
