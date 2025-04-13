import logging
from pathlib import Path
from typing import Any, Dict

import rpy2.robjects as ro
from pydantic import Field
from pydantic.dataclasses import dataclass
from rpy2.rinterface_lib.embedded import RRuntimeError

from components.functional_analysis.base import FunctionalAnalysisBase
from components.functional_analysis.orgdb import OrgDB
from r_wrappers.reactome_pa import enrich_reactome, gsea_reactome


class Config:
    """Configuration class for Pydantic dataclasses.

    This class enables the use of arbitrary types in dataclasses
    decorated with @dataclass(config=Config).
    """

    arbitrary_types_allowed = True


@dataclass(config=Config)
class REACTOMEora(FunctionalAnalysisBase):
    """
    Over-representation analysis for REACTOME Pathways.

    This class implements Over-Representation Analysis (ORA) for Reactome pathways.
    It identifies enriched Reactome pathways in a set of genes of interest compared
    to a background gene set.

    Args:
        background_genes: All genes considered for the experiment.
        org_db: Organism database object for annotation.
        filtered_genes: Genes of interest (e.g., differentially expressed genes).
        files_prefix: Path prefix for output files.
        plots_prefix: Path prefix for output plots.
        func_kwargs: Additional arguments for the Reactome enrichment function.
            Common options include:
            - pAdjustMethod: Method for p-value adjustment, e.g., "BH".
            - pvalueCutoff: P-value cutoff for significance.
            - qvalueCutoff: Q-value cutoff for significance.
            - organism: Organism name (e.g., "human", "mouse").

    Attributes:
        func_result: The enrichment analysis result from ReactomePA.
        func_result_df: DataFrame representation of the enrichment result.
    """

    func_kwargs: Dict[str, Any] = Field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Initialize the Reactome over-representation analysis.

        Calls the enrichment function and then calls the parent's __post_init__
        to process the results.
        """
        # 1. Get functional result
        self.func_result = enrich_reactome(
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
        Generate all plots for Reactome over-representation analysis.

        Creates standard ORA plots suitable for Reactome pathway analysis.

        Args:
            **kwargs: Additional arguments for plotting functions.
        """
        self.plot_all_ora(**kwargs)


@dataclass(config=Config)
class REACTOMEgsea(FunctionalAnalysisBase):
    """
    Gene-set Enrichment analysis for REACTOME Pathways.

    This class implements Gene Set Enrichment Analysis (GSEA) for Reactome pathways.
    It identifies enriched Reactome pathways based on the ranking of genes in a gene list.

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
            - organism: Organism name (e.g., "human", "mouse").
            - seed: Random seed for reproducibility.

    Attributes:
        func_result: The GSEA result from ReactomePA.
        func_result_df: DataFrame representation of the GSEA result.
    """

    func_kwargs: Dict[str, Any] = Field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Initialize the Reactome gene set enrichment analysis.

        Calls the GSEA function and then calls the parent's __post_init__
        to process the results.
        """
        # 1. Get functional result
        self.func_result = gsea_reactome(self.background_genes, **self.func_kwargs)
        super().__post_init__()

    def plot_all(self, **kwargs: Any) -> None:
        """
        Generate all plots for Reactome GSEA.

        Creates standard GSEA plots suitable for Reactome pathway analysis.

        Args:
            **kwargs: Additional arguments for plotting functions.
        """
        self.plot_all_gsea(**kwargs)


def run_reactome_ora(
    background_genes: ro.FloatVector,
    org_db: OrgDB,
    filtered_genes: ro.FloatVector,
    files_prefix: Path,
    plots_prefix: Path,
) -> None:
    """
    Run Reactome over-representation analysis and save results.

    This function creates a REACTOMEora object, runs the analysis,
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
        ora = REACTOMEora(
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


def run_reactome_gsea(
    background_genes: ro.FloatVector,
    org_db: OrgDB,
    filtered_genes: ro.FloatVector,
    files_prefix: Path,
    plots_prefix: Path,
) -> None:
    """
    Run Reactome gene set enrichment analysis and save results.

    This function creates a REACTOMEgsea object, runs the analysis,
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
        gsea = REACTOMEgsea(
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
