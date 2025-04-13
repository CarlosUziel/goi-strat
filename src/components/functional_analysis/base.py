import logging
from pathlib import Path
from typing import Any, Optional

import rpy2.robjects as ro
from pydantic.dataclasses import dataclass
from rpy2.rinterface_lib.embedded import RRuntimeError

from components.functional_analysis.orgdb import OrgDB
from data.utils import TimeoutException, time_limit
from r_wrappers.dose import set_readable
from r_wrappers.enrich_plot import (
    barplot,
    dotplot,
    emapplot,
    gene_concept_net,
    goplot,
    gseaplot,
    heatplot,
    pmcplot,
    ridgeplot,
    upsetplot,
)
from r_wrappers.pathview import pathview
from r_wrappers.utils import rpy2_df_to_pd_df, save_rds


class Config:
    """Configuration class for Pydantic dataclasses.

    This class enables the use of arbitrary types in dataclasses
    decorated with @dataclass(config=Config).
    """

    arbitrary_types_allowed = True


@dataclass(config=Config)
class FunctionalAnalysisBase:
    """
    Base class for functional analyses of gene lists.

    This class provides common functionality for over-representation analysis (ORA)
    and gene set enrichment analysis (GSEA) approaches. It includes methods for
    saving results and creating various visualization plots.

    Args:
        background_genes: All genes considered for the experiment, typically as ranked gene list.
        org_db: Organism database object containing annotation data.
        filtered_genes: Differentially expressed genes, filtered by user criteria.
        files_prefix: Path prefix for all generated data files.
        plots_prefix: Path prefix for all generated plot files.

    Attributes:
        func_result: Raw functional analysis result from R.
        func_result_df: DataFrame representation of the functional analysis result.
    """

    background_genes: ro.FloatVector
    org_db: OrgDB
    filtered_genes: Optional[ro.FloatVector] = None
    files_prefix: Path = Path("")
    plots_prefix: Path = Path("")

    def __post_init__(self) -> None:
        """
        Post-initialization processing.

        Converts functional analysis results to readable format and creates necessary directories.
        Should be called by child classes after their initialization.
        """
        # 1. Get dataframe of result
        # TODO: fix this
        try:
            self.func_result = set_readable(
                self.func_result, self.org_db, keyType="ENTREZID"
            )
            self.func_result_df = rpy2_df_to_pd_df(self.func_result)
        except RRuntimeError as e:
            logging.warning(e)
            self.func_result_df = None

        # 2. Create paths
        self.files_prefix.parent.mkdir(exist_ok=True, parents=True)
        self.plots_prefix.parent.mkdir(exist_ok=True, parents=True)

    def save_rds(self) -> None:
        """
        Save functional analysis result as R data file (.RDS).

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
        Save functional analysis result as CSV file.

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
        Save functional analysis results in all available formats (RDS and CSV).
        """
        self.save_rds()
        self.save_csv()

    def barplot(self, **kwargs: Any) -> None:
        """
        Create a bar plot visualizing enriched terms.

        This plot depicts the enrichment scores (e.g. p-values) color-coded and
        gene count or ratio as bar height.

        Args:
            **kwargs: Additional arguments passed to the R barplot function.
                Common options include:
                - showCategory: Number of categories to show (default: 10)
                - x: Value for x-axis, either "Count" or "GeneRatio"
        """
        if not (self.func_result_df is None or self.func_result_df.empty):
            try:
                save_path = Path(f"{self.plots_prefix}_barplot.pdf")
                barplot(self.func_result, save_path, **kwargs)
            except RRuntimeError as e:
                logging.warning(
                    f"[{self.plots_prefix.name}] Error plotting barplot: \n\t{e}"
                )
        else:
            logging.warning(
                f"[{self.plots_prefix.name}] Could not plot barplot. "
                "Functional result is None or empty."
            )

    def dotplot(self, **kwargs: Any) -> None:
        """
        Create a dot plot visualizing enriched terms.

        Similar to bar plot with the capability to encode another score as dot size.

        Args:
            **kwargs: Additional arguments passed to the R dotplot function.
                Common options include:
                - showCategory: Number of categories to show (default: 10)
                - x: Value for x-axis, either "Count" or "GeneRatio"
        """
        if not (self.func_result_df is None or self.func_result_df.empty):
            try:
                save_path = Path(f"{self.plots_prefix}_dotplot.pdf")
                dotplot(self.func_result, save_path, **kwargs)
            except RRuntimeError as e:
                logging.warning(
                    f"[{self.plots_prefix.name}]Error plotting dotplot: \n\t{e}"
                )
        else:
            logging.warning(
                f"[{self.plots_prefix.name}] Could not plot dotplot. "
                "Functional result is None or empty."
            )

    def gene_concept_net(self, **kwargs: Any) -> None:
        """
        Create a gene-concept network visualization.

        Depicts the linkages of genes and biological concepts (e.g., GO terms or KEGG pathways)
        as a network. Handles both ORA and GSEA results, with only core enriched genes displayed
        for GSEA results.

        Args:
            **kwargs: Additional arguments passed to the R gene_concept_net function.
                Common options include:
                - foldChange: Gene fold changes vector for coloring nodes
                - categorySize: Control the node size of enriched terms
        """
        if not (self.func_result_df is None or self.func_result_df.empty):
            try:
                save_path = Path(f"{self.plots_prefix}_cnetplot.pdf")
                gene_concept_net(
                    self.func_result,
                    save_path,
                    **kwargs,
                )
            except RRuntimeError as e:
                logging.warning(
                    f"[{self.plots_prefix.name}] Error plotting gene concept network: "
                    f"\n\t{e}"
                )
        else:
            logging.warning(
                f"[{self.plots_prefix.name}] Could not plot gene concept network. "
                "Functional result is None or empty."
            )

    def heatplot(self, **kwargs: Any) -> None:
        """
        Create a heatmap visualization of gene-concept relationships.

        Similar to gene concept network, but displays relationships as a heatmap.
        Useful for visualizing large numbers of significant terms more clearly.

        Args:
            **kwargs: Additional arguments passed to the R heatplot function.
                Common options include:
                - foldChange: Gene fold changes vector for coloring
                - showCategory: Number of categories to show
        """
        if not (self.func_result_df is None or self.func_result_df.empty):
            try:
                save_path = Path(f"{self.plots_prefix}_heatplot.pdf")
                heatplot(
                    self.func_result,
                    save_path,
                    **kwargs,
                )
            except RRuntimeError as e:
                logging.warning(
                    f"[{self.plots_prefix.name}] Error plotting heatpot: \n\t{e}"
                )
        else:
            logging.warning(
                f"[{self.plots_prefix.name}] Could not plot heatplot. "
                "Functional result is None or empty."
            )

    def emapplot(self, **kwargs: Any) -> None:
        """
        Create an enrichment map visualization.

        Organizes enriched terms into a network with edges connecting overlapping gene sets.
        Mutually overlapping gene sets cluster together, making it easy to identify functional modules.

        Args:
            **kwargs: Additional arguments passed to the R emapplot function.
                Common options include:
                - layout: Layout algorithm for network visualization
                - showCategory: Number of categories to show
        """
        if not (self.func_result_df is None or self.func_result_df.empty):
            try:
                save_path = Path(f"{self.plots_prefix}_emapplot.pdf")
                emapplot(self.func_result, save_path, cex_label_category=0.8, **kwargs)
            except RRuntimeError as e:
                logging.warning(
                    f"[{self.plots_prefix.name}] Error plotting enrichment map: \n\t{e}"
                )
        else:
            logging.warning(
                f"[{self.plots_prefix.name}] Could not plot enrichment map. "
                "Functional result is None or empty."
            )

    def upsetplot(self, **kwargs: Any) -> None:
        """
        Create an upset plot visualization.

        Alternative to gene concept network for visualizing complex associations between genes
        and gene sets. Emphasizes gene overlapping among different gene sets.

        For ORA, calculates overlaps among different gene sets. For GSEA results,
        plots the fold change distributions of different categories.

        Args:
            **kwargs: Additional arguments passed to the R upsetplot function.
        """
        if not (self.func_result_df is None or self.func_result_df.empty):
            try:
                save_path = Path(f"{self.plots_prefix}_upsetplot.pdf")
                upsetplot(self.func_result, save_path, **kwargs)
            except RRuntimeError as e:
                logging.warning(
                    f"[{self.plots_prefix.name}] Error plotting upsetplot: \n\t{e}"
                )
        else:
            logging.warning(
                f"[{self.plots_prefix.name}] Could not plot upsetplot. "
                "Functional result is None or empty."
            )

    def ridgeplot(self, **kwargs: Any) -> None:
        """
        Create a ridge plot visualization.

        Visualizes expression distributions of core enriched genes for GSEA enriched categories.
        Helps users to interpret up/down-regulated pathways.

        Args:
            **kwargs: Additional arguments passed to the R ridgeplot function.
                Common options include:
                - showCategory: Number of categories to show
                - fill: Color scheme for filling ridges
        """
        if not (self.func_result_df is None or self.func_result_df.empty):
            try:
                save_path = Path(f"{self.plots_prefix}_ridgeplot.pdf")
                ridgeplot(self.func_result, save_path, **kwargs)
            except RRuntimeError as e:
                logging.warning(
                    f"[{self.plots_prefix.name}] Error plotting ridgeplot: \n\t{e}"
                )
        else:
            logging.warning(
                f"[{self.plots_prefix.name}] Could not plot ridgeplot. "
                "Functional result is None or empty."
            )

    def gseaplot(self, gene_set_id: int = 1, **kwargs: Any) -> None:
        """
        Create a GSEA plot visualization.

        Visualizes running score and preranked gene list for GSEA results.
        Shows the distribution of the gene set and the enrichment score.

        Args:
            gene_set_id: Index of the gene set to visualize (1-based). Default is 1 (first gene set).
            **kwargs: Additional arguments passed to the R gseaplot function.
                Common options include:
                - title: Plot title
                - color: Colors for line and points
        """
        if not (self.func_result_df is None or self.func_result_df.empty):
            try:
                save_path = Path(f"{self.plots_prefix}_gseaplot.pdf")
                gseaplot(self.func_result, gene_set_id, save_path, **kwargs)
            except RRuntimeError as e:
                logging.warning(
                    f"[{self.plots_prefix.name}] Error plotting gseaplot: \n\t{e}"
                )
        else:
            logging.warning(
                f"[{self.plots_prefix.name}] Could not plot gseaplot. "
                "Functional result is None or empty."
            )

    def pmcplot(self, **kwargs: Any) -> None:
        """
        Create a PubMed Central publication trend plot.

        Plots the number/proportion of publications trend based on query results from PubMed Central.
        Uses the top enriched terms as queries.

        Args:
            **kwargs: Additional arguments passed to the R pmcplot function.
                Common options include:
                - by: Either "proportion" or "number" to visualize publication counts
                - from: Start year for the publication trend
                - to: End year for the publication trend
        """
        if not (self.func_result_df is None or self.func_result_df.empty):
            terms = ro.StrVector(self.func_result_df["Description"][:5].tolist())
            try:
                save_path = Path(f"{self.plots_prefix}_pmcplot.pdf")
                pmcplot(terms, save_path, **kwargs)
            except RRuntimeError as e:
                logging.warning(
                    f"[{self.plots_prefix.name}] Error plotting pubmed plot: \n\t{e}"
                )
        else:
            logging.warning(
                f"[{self.plots_prefix.name}] Could not plot pmcplot. "
                "Functional result is None or empty."
            )

    def goplot(self, **kwargs: Any) -> None:
        """
        Create a GO plot visualization.

        Visualizes Gene Ontology enrichment results with hierarchical structures.

        Args:
            **kwargs: Additional arguments passed to the R goplot function.
                Common options include:
                - showCategory: Number of categories to show
                - color: Color for dots and lines
        """
        if not (self.func_result_df is None or self.func_result_df.empty):
            try:
                save_path = Path(f"{self.plots_prefix}_goplot.pdf")
                goplot(self.func_result, save_path, **kwargs)
            except RRuntimeError as e:
                logging.warning(
                    f"[{self.plots_prefix.name}] Error plotting goplot: \n\t{e}"
                )
        else:
            logging.warning(
                f"[{self.plots_prefix.name}] Could not plot goplot. "
                "Functional result is None or empty."
            )

    def pathview(self, gene_data: ro.FloatVector, **kwargs: Any) -> None:
        """
        Map and render enriched pathways using Pathview.

        Creates pathway visualizations with gene expression data overlaid on KEGG pathway maps.

        Args:
            gene_data: Gene expression data to overlay on pathways.
            **kwargs: Additional arguments passed to the R pathview function.
                Common options include:
                - species: KEGG species code (e.g., "hsa" for human)
                - kegg.dir: Directory to store KEGG pathway data
        """
        if not (self.func_result_df is None or self.func_result_df.empty):
            save_path_root = Path(f"{self.plots_prefix}_pathview")
            save_path_root.mkdir(exist_ok=True, parents=True)
            try:
                for pathway_id, pathway_name in self.func_result_df[
                    ["ID", "Description"]
                ].itertuples(index=False, name=None):
                    try:
                        with time_limit(60):
                            pathview(
                                gene_data,
                                pathway_id,
                                pathway_name,
                                save_path_root,
                                **kwargs,
                            )
                    except TimeoutException:
                        logging.warning(
                            f"[{self.plots_prefix.name}] Time out running pathview."
                        )
            except RRuntimeError as e:
                logging.warning(
                    f"[{self.plots_prefix.name}] Error plotting with pathview: \n\t{e}"
                )
        else:
            logging.warning(
                f"[{self.plots_prefix.name}] Could not plot with pathview. "
                "Functional result is None or empty."
            )

    def plot_all_common(self, **kwargs: Any) -> None:
        """
        Generate a set of common plots suitable for all analysis types.

        Creates dotplot, emapplot, upsetplot, and pmcplot.

        Args:
            **kwargs: Additional arguments passed to individual plotting functions.
        """
        self.dotplot(showCategory=10, x="Count", **kwargs)
        self.emapplot(**kwargs)
        self.upsetplot(**kwargs)
        self.pmcplot(**kwargs)

    def plot_all_ora(self, **kwargs: Any) -> None:
        """
        Generate all plots suitable for Over-Representation Analysis (ORA).

        Creates barplot, common plots, gene-concept network, and heatplot.

        Args:
            **kwargs: Additional arguments passed to individual plotting functions.
        """
        self.barplot(showCategory=10, x="Count", **kwargs)
        self.plot_all_common(**kwargs)
        self.gene_concept_net(**kwargs, foldChange=self.filtered_genes)
        self.heatplot(**kwargs, foldChange=self.filtered_genes)

    def plot_all_gsea(self, **kwargs: Any) -> None:
        """
        Generate all plots suitable for Gene Set Enrichment Analysis (GSEA).

        Creates common plots, gene-concept network, heatplot, ridgeplot, and gseaplot.

        Args:
            **kwargs: Additional arguments passed to individual plotting functions.
        """
        self.plot_all_common(**kwargs)
        self.gene_concept_net(**kwargs, foldChange=self.background_genes)
        self.heatplot(**kwargs, foldChange=self.background_genes)
        self.ridgeplot(**kwargs)
        self.gseaplot(**kwargs)
