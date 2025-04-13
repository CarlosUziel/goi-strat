import logging
from pathlib import Path
from typing import Any, Dict

import rpy2.robjects as ro
from pydantic import Field
from pydantic.dataclasses import dataclass
from rpy2.rinterface_lib.embedded import RRuntimeError

from components.functional_analysis.base import FunctionalAnalysisBase
from components.functional_analysis.orgdb import OrgDB
from r_wrappers.dose import enrich_do, gse_do


class Config:
    """Configuration class for Pydantic dataclasses.

    This class enables the use of arbitrary types in dataclasses
    decorated with @dataclass(config=Config).
    """

    arbitrary_types_allowed = True


@dataclass(config=Config)
class DOora(FunctionalAnalysisBase):
    """
    Over-representation analysis for Disease Ontology (DO).

    This class implements Over-Representation Analysis (ORA) for Disease Ontology terms.
    It identifies enriched disease terms in a set of genes of interest compared
    to a background gene set. Disease Ontology (DO) is a standardized ontology
    for human disease terms with a focus on connecting genes to diseases.

    Args:
        background_genes: All genes considered for the experiment.
        org_db: Organism database object for annotation.
        filtered_genes: Genes of interest (e.g., differentially expressed genes).
        files_prefix: Path prefix for output files.
        plots_prefix: Path prefix for output plots.
        func_kwargs: Additional arguments for the DO enrichment function.
            Common options include:
            - pAdjustMethod: Method for p-value adjustment, e.g., "BH".
            - pvalueCutoff: P-value cutoff for significance.
            - qvalueCutoff: Q-value cutoff for significance.
            - minGSSize: Minimum gene set size.
            - maxGSSize: Maximum gene set size.

    Attributes:
        func_result: The enrichment analysis result from DOSE package.
        func_result_df: DataFrame representation of the enrichment result.
    """

    func_kwargs: Dict[str, Any] = Field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Initialize the Disease Ontology over-representation analysis.

        Calls the enrichment function and then calls the parent's __post_init__
        to process the results.
        """
        # 1. Get functional result
        self.func_result = enrich_do(
            self.filtered_genes.names,
            universe=self.background_genes.names,
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
        Generate all plots for Disease Ontology over-representation analysis.

        Creates standard ORA plots suitable for Disease Ontology analysis.

        Args:
            **kwargs: Additional arguments for plotting functions.
        """
        self.plot_all_ora(**kwargs)


@dataclass(config=Config)
class DOgsea(FunctionalAnalysisBase):
    """
    Gene-set Enrichment analysis for Disease Ontology (DO).

    This class implements Gene Set Enrichment Analysis (GSEA) for Disease Ontology terms.
    It identifies enriched disease terms based on the ranking of genes in a gene list.
    Disease Ontology (DO) is a standardized ontology for human disease terms
    with a focus on connecting genes to diseases.

    Args:
        background_genes: Ranked gene list with scores (e.g., log fold changes).
        org_db: Organism database object for annotation.
        filtered_genes: Optional subset of genes of interest.
        files_prefix: Path prefix for output files.
        plots_prefix: Path prefix for output plots.
        func_kwargs: Additional arguments for the GSEA function.
            Common options include:
            - pAdjustMethod: Method for p-value adjustment, e.g., "BH".
            - minGSSize: Minimum gene set size.
            - maxGSSize: Maximum gene set size.
            - pvalueCutoff: P-value cutoff for significance.
            - nPerm: Number of permutations.
            - seed: Random seed for reproducibility.

    Attributes:
        func_result: The GSEA result from DOSE package.
        func_result_df: DataFrame representation of the GSEA result.
    """

    func_kwargs: Dict[str, Any] = Field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Initialize the Disease Ontology gene set enrichment analysis.

        Calls the GSEA function and then calls the parent's __post_init__
        to process the results.
        """
        # 1. Get functional result
        self.func_result = gse_do(self.background_genes, **self.func_kwargs)
        super().__post_init__()

    def plot_all(self, **kwargs: Any) -> None:
        """
        Generate all plots for Disease Ontology GSEA.

        Creates standard GSEA plots suitable for Disease Ontology analysis.

        Args:
            **kwargs: Additional arguments for plotting functions.
        """
        self.plot_all_gsea(**kwargs)


def run_do_ora(
    background_genes: ro.FloatVector,
    org_db: OrgDB,
    filtered_genes: ro.FloatVector,
    files_prefix: Path,
    plots_prefix: Path,
) -> None:
    """
    Run Disease Ontology over-representation analysis and save results.

    This function creates a DOora object, runs the analysis,
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
        ora = DOora(
            background_genes,
            org_db,
            filtered_genes,
            files_prefix,
            plots_prefix,
        )
        ora.save_csv()
        ora.plot_all()
    except RRuntimeError as e:
        logging.warning(
            f"[{plots_prefix.name}] Error computing functional result: \n\t{e}"
        )


def run_do_gsea(
    background_genes: ro.FloatVector,
    org_db: OrgDB,
    filtered_genes: ro.FloatVector,
    files_prefix: Path,
    plots_prefix: Path,
) -> None:
    """
    Run Disease Ontology gene set enrichment analysis and save results.

    This function creates a DOgsea object, runs the analysis,
    saves the results as CSV, and generates plots.

    Args:
        background_genes: Ranked gene list with scores (e.g., log fold changes).
        org_db: Organism database object for annotation.
        filtered_genes: Optional subset of genes of interest.
        files_prefix: Path prefix for output files.
        plots_prefix: Path prefix for output plots.

    Returns:
        None. Results are saved to files.
    """
    try:
        gsea = DOgsea(
            background_genes,
            org_db,
            filtered_genes,
            files_prefix,
            plots_prefix,
        )
        gsea.save_csv()
        gsea.plot_all()
    except RRuntimeError as e:
        logging.warning(
            f"[{plots_prefix.name}] Error computing functional result: \n\t{e}"
        )
