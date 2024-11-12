import logging
from pathlib import Path
from typing import Optional

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
    arbitrary_types_allowed = True


@dataclass(config=Config)
class FunctionalAnalysisBase:
    """
    Base class for functional analyses of gene lists.

    Args:
        background_genes: All genes considered for the experiment.
        filtered_genes: Differentially expressed genes, filtered by user criteria.
        files_prefix: Path to be prefixed to all generated files.
        plots_prefix: Path to be prefixed to all generated files.
    """

    background_genes: ro.FloatVector
    org_db: OrgDB
    filtered_genes: Optional[ro.FloatVector] = None
    files_prefix: Path = Path("")
    plots_prefix: Path = Path("")

    def __post_init__(self):
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

    def save_rds(self):
        if not (self.func_result_df is None or self.func_result_df.empty):
            save_rds(self.func_result, self.files_prefix.with_suffix(".RDS"))
        else:
            logging.warning(
                f"[{self.plots_prefix.name}] Could not save RDS. "
                "Functional result is None or empty."
            )

    def save_csv(self):
        if not (self.func_result_df is None or self.func_result_df.empty):
            self.func_result_df.to_csv(self.files_prefix.with_suffix(".csv"))
        else:
            logging.warning(
                f"[{self.plots_prefix.name}] Could not save CSV. "
                "Functional result is None or empty."
            )

    def save_all(self):
        self.save_rds()
        self.save_csv()

    def barplot(self, **kwargs):
        """
        Bar plot is the most widely used method to visualize enriched terms.
        It depicts the enrichment scores (e.g. p values) color-coded and
        gene count or ratio as bar height.
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

    def dotplot(self, **kwargs):
        """
        Dot plot is similar to bar plot with the capability to encode another
        score as dot size.
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

    def gene_concept_net(self, **kwargs):
        """
        Both the barplot and dotplot only displayed most significant
        enriched terms, while users may want to know which genes are
        involved in these significant terms. In order to consider the
        potentially biological complexities in which a gene may belong to
        multiple annotation categories and provide information of numeric
        changes if available, we developed cnetplot function to extract the
        complex association. The cnetplot depicts the linkages of genes and
        biological concepts (e.g. GO terms or KEGG pathways) as a network.
        GSEA result is also supported with only core enriched genes displayed.
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

    def heatplot(self, **kwargs):
        """
        The heatplot is similar to cnetplot, while displaying the
        relationships as a heatmap. The gene-concept network may become too
        complicated if user want to show a large number significant terms.
        The heatplot can simplify the result and more easy to identify
        expression patterns.
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

    def emapplot(self, **kwargs):
        """
        Enrichment map organizes enriched terms into a network with edges
        connecting overlapping gene sets. In this way, mutually overlapping
        gene sets tend to cluster together, making it easy to identify
        functional module.
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

    def upsetplot(self, **kwargs):
        """
        The upsetplot is an alternative to cnetplot for visualizing the
        complex association between genes and gene sets. It emphasizes the
        gene overlapping among different gene sets.

        For over-representation analysis, upsetplot will calculate the
        overlaps among different gene sets. For GSEA result, it will plot the
        fold change distributions of different categories (e.g. unique to
        pathway, overlaps among different pathways).
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

    def ridgeplot(self, **kwargs):
        """
        The ridgeplot will visualize expression distributions of core
        enriched genes for GSEA enriched categories. It helps users to
        interpret up/down-regulated pathways.
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

    def gseaplot(self, gene_set_id: int = 1, **kwargs):
        """
        Running score and preranked list are traditional methods for
        visualizing GSEA result. The enrichplot package supports both of
        them to visualize the distribution of the gene set and the
        enrichment score.
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

    def pmcplot(self, **kwargs):
        """
        One of the problem of enrichment analysis is to find pathways for
        further investigation. Here, we provide pmcplot function to plot the
        number/proportion of publications trend based on the query result
        from PubMed Central. Of course, users can use pmcplot in other
        scenarios. All text that can be queried on PMC is valid as input of
        pmcplot.
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

    def goplot(self, **kwargs):
        """
        Bar plot is the most widely used method to visualize enriched terms.
        It depicts the enrichment scores (e.g. p values) color-coded and
        gene count or ratio as bar height.
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

    def pathview(self, gene_data: ro.FloatVector, **kwargs):
        """
        Map and render enriched pathways.
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

    def plot_all_common(self, **kwargs):
        self.dotplot(showCategory=10, x="Count", **kwargs)
        self.emapplot(**kwargs)
        self.upsetplot(**kwargs)
        self.pmcplot(**kwargs)

    def plot_all_ora(self, **kwargs):
        self.barplot(showCategory=10, x="Count", **kwargs)
        self.plot_all_common(**kwargs)
        self.gene_concept_net(**kwargs, foldChange=self.filtered_genes)
        self.heatplot(**kwargs, foldChange=self.filtered_genes)

    def plot_all_gsea(self, **kwargs):
        self.plot_all_common(**kwargs)
        self.gene_concept_net(**kwargs, foldChange=self.background_genes)
        self.heatplot(**kwargs, foldChange=self.background_genes)
        self.ridgeplot(**kwargs)
        self.gseaplot(**kwargs)
