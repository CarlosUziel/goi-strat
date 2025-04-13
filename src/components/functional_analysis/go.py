import logging
from pathlib import Path
from typing import Any, Dict

import rpy2.robjects as ro
from pydantic import Field
from pydantic.dataclasses import dataclass
from rpy2.rinterface_lib.embedded import RRuntimeError

from components.functional_analysis.base import FunctionalAnalysisBase
from components.functional_analysis.orgdb import OrgDB
from r_wrappers.cluster_profiler import enrich_go, gse_go


class Config:
    """Configuration class for Pydantic dataclasses.

    This class enables the use of arbitrary types in dataclasses
    decorated with @dataclass(config=Config).
    """

    arbitrary_types_allowed = True


@dataclass(config=Config)
class GOora(FunctionalAnalysisBase):
    """
    Over-representation analysis for Gene Ontology terms.

    This class implements Over-Representation Analysis (ORA) for Gene Ontology (GO) terms.
    It identifies enriched GO terms in a set of genes of interest compared to a background gene set.

    Args:
        background_genes: All genes considered for the experiment.
        org_db: Organism database object for annotation.
        filtered_genes: Genes of interest (e.g., differentially expressed genes).
        files_prefix: Path prefix for output files.
        plots_prefix: Path prefix for output plots.
        func_kwargs: Additional arguments for the GO enrichment function.
            Common options include:
            - ont: GO ontology to analyze ("BP", "MF", "CC", or "ALL").
            - pAdjustMethod: Method for p-value adjustment, e.g., "BH".
            - pvalueCutoff: P-value cutoff for significance.
            - qvalueCutoff: Q-value cutoff for significance.

    Attributes:
        func_result: The enrichment analysis result from clusterProfiler.
        func_result_df: DataFrame representation of the enrichment result.
    """

    func_kwargs: Dict[str, Any] = Field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Initialize the Gene Ontology over-representation analysis.

        Calls the enrichment function and then calls the parent's __post_init__
        to process the results.
        """
        # 1. Get functional result
        self.func_result = enrich_go(
            self.filtered_genes.names,
            universe=self.background_genes.names,
            org_db=self.org_db,
            **self.func_kwargs,
        )
        super().__post_init__()

    def ridgeplot(self, **kwargs: Any) -> None:
        """
        Not implemented for over-representation analysis.

        Raises:
            NotImplementedError: ORA results cannot be used for GSEA plots.
        """
        raise NotImplementedError("ORA result cannot be used for GSEA plots.")

    def gseaplot(self, gene_set_id: int, **kwargs: Any) -> None:
        """
        Not implemented for over-representation analysis.

        Args:
            gene_set_id: The index of the gene set to plot.
            **kwargs: Additional arguments for plotting.

        Raises:
            NotImplementedError: ORA results cannot be used for GSEA plots.
        """
        raise NotImplementedError("ORA result cannot be used for GSEA plots.")

    def plot_all_gsea(self, **kwargs: Any) -> None:
        """
        Not implemented for over-representation analysis.

        Args:
            **kwargs: Additional arguments for plotting.

        Raises:
            NotImplementedError: ORA results cannot be used for GSEA plots.
        """
        raise NotImplementedError("ORA result cannot be used for GSEA plots.")

    def plot_all(self, **kwargs: Any) -> None:
        """
        Generate all plots for GO over-representation analysis.

        Creates a GO-specific plot and standard ORA plots.

        Args:
            **kwargs: Additional arguments for plotting functions.
        """
        self.goplot(**kwargs)
        self.plot_all_ora(**kwargs)


@dataclass(config=Config)
class GOgsea(FunctionalAnalysisBase):
    """
    Gene-set Enrichment analysis for Gene Ontology terms.

    This class implements Gene Set Enrichment Analysis (GSEA) for Gene Ontology (GO) terms.
    It identifies enriched GO terms based on the ranking of genes in a gene list.

    Args:
        background_genes: Ranked gene list with scores (e.g., log fold changes).
        org_db: Organism database object for annotation.
        filtered_genes: Optional subset of genes of interest.
        files_prefix: Path prefix for output files.
        plots_prefix: Path prefix for output plots.
        func_kwargs: Additional arguments for the GSEA function.
            Common options include:
            - ont: GO ontology to analyze ("BP", "MF", "CC", or "ALL").
            - pAdjustMethod: Method for p-value adjustment, e.g., "BH".
            - minGSSize: Minimum gene set size.
            - maxGSSize: Maximum gene set size.
            - pvalueCutoff: P-value cutoff for significance.
            - seed: Random seed for reproducibility.

    Attributes:
        func_result: The GSEA result from clusterProfiler.
        func_result_df: DataFrame representation of the GSEA result.
    """

    func_kwargs: Dict[str, Any] = Field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Initialize the Gene Ontology gene set enrichment analysis.

        Calls the GSEA function and then calls the parent's __post_init__
        to process the results.
        """
        # 1. Get functional result
        self.func_result = gse_go(
            self.background_genes, org_db=self.org_db, **self.func_kwargs
        )
        super().__post_init__()

    def plot_all(self, **kwargs: Any) -> None:
        """
        Generate all plots for Gene Ontology GSEA.

        Creates standard GSEA plots suitable for GO analysis.

        Args:
            **kwargs: Additional arguments for plotting functions.
        """
        self.plot_all_gsea(**kwargs)


def run_go_ora(
    background_genes: ro.FloatVector,
    org_db: OrgDB,
    filtered_genes: ro.FloatVector,
    files_prefix: Path,
    plots_prefix: Path,
    ont: str,
) -> None:
    """
    Run Gene Ontology over-representation analysis and save results.

    This function creates a GOora object, runs the analysis for the specified
    GO ontology, saves the results as CSV, and generates plots.

    Args:
        background_genes: All genes considered for the experiment.
        org_db: Organism database object for annotation.
        filtered_genes: Genes of interest (e.g., differentially expressed genes).
        files_prefix: Path prefix for output files.
        plots_prefix: Path prefix for output plots.
        ont: GO ontology to analyze ("BP" for biological process, "MF" for
            molecular function, "CC" for cellular component, or "ALL" for all ontologies).

    Returns:
        None. Results are saved to files.
    """
    try:
        ora = GOora(
            background_genes,
            org_db,
            filtered_genes,
            files_prefix,
            plots_prefix,
            {"ont": ont},
        )
        ora.save_csv()
        ora.plot_all()
    except RRuntimeError as e:
        logging.warning(
            f"[{plots_prefix.name}] Error computing functional result: \n\t{e}"
        )


def run_go_gsea(
    background_genes: ro.FloatVector,
    org_db: OrgDB,
    filtered_genes: ro.FloatVector,
    files_prefix: Path,
    plots_prefix: Path,
    ont: str,
) -> None:
    """
    Run Gene Ontology gene set enrichment analysis and save results.

    This function creates a GOgsea object, runs the analysis for the specified
    GO ontology, saves the results as CSV, and generates plots.

    Args:
        background_genes: Ranked gene list with scores (e.g., log fold changes).
        org_db: Organism database object for annotation.
        filtered_genes: Optional subset of genes of interest.
        files_prefix: Path prefix for output files.
        plots_prefix: Path prefix for output plots.
        ont: GO ontology to analyze ("BP" for biological process, "MF" for
            molecular function, "CC" for cellular component, or "ALL" for all ontologies).

    Returns:
        None. Results are saved to files.
    """
    try:
        gsea = GOgsea(
            background_genes,
            org_db,
            filtered_genes,
            files_prefix,
            plots_prefix,
            {"ont": ont},
        )
        gsea.save_csv()
        gsea.plot_all()
    except RRuntimeError as e:
        logging.warning(
            f"[{plots_prefix.name}] Error computing functional result: \n\t{e}"
        )
