import logging
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Optional

import rpy2.robjects as ro
from pydantic import Field
from pydantic.dataclasses import dataclass
from rpy2.rinterface_lib.embedded import RRuntimeError

from components.functional_analysis.orgdb import OrgDB
from r_wrappers.gprofiler2 import gost, gost_plot, publish_gost_plot, publish_gost_table
from r_wrappers.utils import map_gene_id, rpy2_df_to_pd_df_manual, save_rds


class Config:
    """Configuration class for Pydantic dataclasses.

    This class enables the use of arbitrary types in dataclasses
    decorated with @dataclass(config=Config).
    """

    arbitrary_types_allowed = True


@dataclass(config=Config)
class GPROFILER2base:
    """
    Base class for enrichment analysis of multiple gene sets using gprofiler2.

    g:Profiler is a web tool and R package for functional enrichment analysis and
    conversions of gene lists. This class provides a Python interface to g:Profiler
    functionality through the R gprofiler2 package.

    Args:
        background_genes: All genes considered for the experiment.
        org_db: Organism database object for annotation.
        filtered_genes: Differentially expressed genes, filtered by user criteria.
        files_prefix: Path prefix for output files.
        plots_prefix: Path prefix for output plots.
        func_kwargs: Additional arguments for functional analysis function.

    Attributes:
        func_result: Raw result from g:Profiler analysis.
        func_result_df: DataFrame representation of the enrichment result.
    """

    background_genes: ro.FloatVector
    org_db: OrgDB
    filtered_genes: Optional[ro.FloatVector] = None
    files_prefix: Path = Path("")
    plots_prefix: Path = Path("")
    func_kwargs: Dict[str, Any] = Field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Post-initialization processing.

        Converts g:Profiler results to a pandas DataFrame and processes gene IDs to symbols.
        Creates output directories for files and plots.
        """
        # 1. Get dataframe of result
        func_result = getattr(self, "func_result", None)
        try:
            self.func_result_df = (
                rpy2_df_to_pd_df_manual(func_result.rx2("result"))
                .astype({"p_value": "float"})
                .sort_values("p_value")
            )
            if "intersection" in self.func_result_df.columns:
                symbol_map = map_gene_id(
                    chain(
                        *[
                            gene_list.split(",")
                            for gene_list in self.func_result_df["intersection"]
                        ]
                    ),
                    self.org_db,
                    "ENTREZID",
                    "SYMBOL",
                ).to_dict()

                self.func_result_df["intersection"] = [
                    "/".join([symbol_map[gene] for gene in gene_list.split(",")])
                    for gene_list in self.func_result_df["intersection"]
                ]
        except Exception as e:
            logging.warning(e)
            self.func_result_df = None

        # 2. Create paths
        self.files_prefix.parent.mkdir(exist_ok=True, parents=True)
        self.plots_prefix.parent.mkdir(exist_ok=True, parents=True)

    def save_rds(self) -> None:
        """
        Save g:Profiler analysis result as an R data file (.RDS).

        The file will be saved with the path specified by files_prefix with .RDS extension.
        """
        if not (self.func_result_df is None or self.func_result_df.empty):
            save_rds(self.func_result, self.files_prefix.with_suffix(".RDS"))
        else:
            logging.warning(
                f"[{self.plots_prefix.name}] Could not save RDS. "
                "Functional result is None or empty."
            )

    def save_csv(self) -> None:
        """
        Save g:Profiler analysis result as a CSV file.

        The file will be saved with the path specified by files_prefix with .csv extension.
        """
        if not (self.func_result_df is None or self.func_result_df.empty):
            self.func_result_df.to_csv(self.files_prefix.with_suffix(".csv"))
        else:
            logging.warning(
                f"[{self.plots_prefix.name}] Could not save CSV. "
                "Functional result is None or empty."
            )

    def save_all(self) -> None:
        """
        Save g:Profiler analysis results in all available formats (RDS and CSV).
        """
        self.save_rds()
        self.save_csv()

    def plot(self, **kwargs: Any) -> None:
        """
        Create a basic g:Profiler enrichment plot.

        Visualizes enrichment results in a static (non-interactive) format.

        Args:
            **kwargs: Additional arguments passed to the gost_plot function.
                Common options include:
                - interactive: Whether to create an interactive plot (always set to False here)
                - capped: Whether to cap very small p-values
        """
        if not (self.func_result_df is None or self.func_result_df.empty):
            try:
                save_path = Path(f"{self.plots_prefix}_plot.pdf")
                gost_plot(self.func_result, save_path, interactive=False, **kwargs)
            except RRuntimeError as e:
                logging.warning(
                    f"[{self.plots_prefix.name}] Error plotting results: \n\t{e}"
                )
        else:
            logging.warning(
                f"[{self.plots_prefix.name}] Could not plot results. "
                "Functional result is None or empty."
            )

    def tableplot(self, **kwargs: Any) -> None:
        """
        Create a table visualization of g:Profiler results.

        Generates a styled table for publication with enrichment results.

        Args:
            **kwargs: Additional arguments passed to the publish_gost_table function.
        """
        if not (self.func_result_df is None or self.func_result_df.empty):
            try:
                save_path = Path(f"{self.plots_prefix}_tableplot.pdf")
                publish_gost_table(self.func_result, save_path, **kwargs)
            except RRuntimeError as e:
                logging.warning(
                    f"[{self.plots_prefix.name}] Error plotting table plot: \n\t{e}"
                )
        else:
            logging.warning(
                f"[{self.plots_prefix.name}] Could plot table plot. "
                "Functional result is None or empty."
            )

    def gostplot(self, **kwargs: Any) -> None:
        """
        Create a publication-ready g:Profiler Manhattan plot.

        Generates an advanced visualization highlighting the top enriched terms.

        Args:
            **kwargs: Additional arguments passed to the publish_gost_plot function.
                Common options include:
                - highlight_terms: Vector of term IDs to highlight (top 10 by default)
                - filename: Output filename
        """
        if not (self.func_result_df is None or self.func_result_df.empty):
            try:
                save_path = Path(f"{self.plots_prefix}_gostplot.pdf")
                top_terms = ro.StrVector(list(self.func_result_df["term_id"][:10]))
                publish_gost_plot(
                    gost_plot(self.func_result, interactive=False),
                    save_path,
                    highlight_terms=top_terms,
                    **kwargs,
                )
            except RRuntimeError as e:
                logging.warning(
                    f"[{self.plots_prefix.name}] Error plotting gost plot: \n\t{e}"
                )
        else:
            logging.warning(
                f"[{self.plots_prefix.name}] Could not plot gostplot. "
                "Functional result is None or empty."
            )

    def plot_all(self, **kwargs: Any) -> None:
        """
        Generate all available g:Profiler plots.

        Creates basic plot, table plot, and Manhattan plot for the enrichment results.

        Args:
            **kwargs: Additional arguments passed to individual plotting functions.
        """
        self.plot(**kwargs)
        self.tableplot(**kwargs)
        self.gostplot(**kwargs)


@dataclass(config=Config)
class GPROFILER2ora(GPROFILER2base):
    """
    Over-representation analysis for multiple gene sets using gprofiler2.

    This class implements Over-Representation Analysis (ORA) using g:Profiler.
    It can analyze multiple gene sets simultaneously for enrichment in various
    functional categories like GO terms, pathways, and regulatory motifs.

    Args:
        background_genes: All genes considered for the experiment.
        org_db: Organism database object for annotation.
        filtered_genes: Genes of interest (e.g., differentially expressed genes).
        files_prefix: Path prefix for output files.
        plots_prefix: Path prefix for output plots.
        func_kwargs: Additional arguments for the g:Profiler enrichment function.
            Common options include:
            - organism: Organism name (e.g., "hsapiens", "mmusculus").
            - sources: Data sources to use (e.g., "GO:BP", "KEGG", "REAC").
            - correction_method: Method for multiple testing correction.
            - user_threshold: Significance threshold.

    Attributes:
        func_result: Raw result from g:Profiler analysis.
        func_result_df: DataFrame representation of the enrichment result.
    """

    func_kwargs: Dict[str, Any] = Field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Initialize the g:Profiler over-representation analysis.

        Calls the g:Profiler gost function and then calls the parent's __post_init__
        to process the results.
        """
        # 1. Get functional result
        self.func_result = gost(
            self.filtered_genes.names,
            custom_bg=self.background_genes.names,
            evcodes=True,
            **self.func_kwargs,
        )
        super().__post_init__()


@dataclass(config=Config)
class GPROFILER2gsea(GPROFILER2base):
    """
    Ordered query analysis for multiple gene sets using gprofiler2.

    This class implements an analysis similar to Gene Set Enrichment Analysis
    using g:Profiler's ordered query functionality. It analyzes a ranked list of genes
    for enrichment in various functional categories.

    Note:
        This is not a true GSEA implementation but uses g:Profiler's ordered query
        functionality which considers the order of genes in the input list.

    Args:
        background_genes: Ranked gene list (order matters for this analysis).
        org_db: Organism database object for annotation.
        filtered_genes: Optional subset of genes of interest.
        files_prefix: Path prefix for output files.
        plots_prefix: Path prefix for output plots.
        func_kwargs: Additional arguments for the g:Profiler function.
            Common options include:
            - organism: Organism name (e.g., "hsapiens", "mmusculus").
            - sources: Data sources to use (e.g., "GO:BP", "KEGG", "REAC").
            - correction_method: Method for multiple testing correction.
            - user_threshold: Significance threshold.

    Attributes:
        func_result: Raw result from g:Profiler analysis.
        func_result_df: DataFrame representation of the enrichment result.
    """

    func_kwargs: Dict[str, Any] = Field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Initialize the g:Profiler ordered query analysis.

        Calls the g:Profiler gost function with ordered_query=True and then calls
        the parent's __post_init__ to process the results.
        """
        # 1. Get functional result
        self.func_result = gost(
            self.background_genes.names,
            evcodes=True,
            ordered_query=True,
            **self.func_kwargs,
        )
        super().__post_init__()


def run_gprofiler2_ora(
    background_genes: ro.FloatVector,
    org_db: OrgDB,
    filtered_genes: ro.FloatVector,
    files_prefix: Path,
    plots_prefix: Path,
) -> None:
    """
    Run g:Profiler over-representation analysis and save results.

    This function creates a GPROFILER2ora object, runs the analysis,
    saves the results as CSV, and generates plots.

    Args:
        background_genes: All genes considered for the experiment.
        org_db: Organism database object for annotation.
        filtered_genes: Genes of interest (e.g., differentially expressed genes).
        files_prefix: Path prefix for output files.
        plots_prefix: Path prefix for output plots.

    Returns:
        None. Results are saved to files.
    """
    try:
        ora = GPROFILER2ora(
            background_genes, org_db, filtered_genes, files_prefix, plots_prefix
        )
        ora.save_csv()
        ora.plot_all()
    except RRuntimeError as e:
        logging.warning(
            f"[{plots_prefix.name}] Error computing functional result: \n\t{e}"
        )


def run_gprofiler2_gsea(
    background_genes: ro.FloatVector,
    org_db: OrgDB,
    filtered_genes: ro.FloatVector,
    files_prefix: Path,
    plots_prefix: Path,
) -> None:
    """
    Run g:Profiler ordered query analysis and save results.

    This function creates a GPROFILER2gsea object, runs the analysis using
    ordered query mode, saves the results as CSV, and generates plots.

    Args:
        background_genes: Ranked gene list (order matters for this analysis).
        org_db: Organism database object for annotation.
        filtered_genes: Optional subset of genes of interest.
        files_prefix: Path prefix for output files.
        plots_prefix: Path prefix for output plots.

    Returns:
        None. Results are saved to files.
    """
    try:
        gsea = GPROFILER2gsea(
            background_genes, org_db, filtered_genes, files_prefix, plots_prefix
        )
        gsea.save_csv()
        gsea.plot_all()
    except RRuntimeError as e:
        logging.warning(
            f"[{plots_prefix.name}] Error computing functional result: \n\t{e}"
        )
