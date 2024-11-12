import logging
from pathlib import Path
from typing import Any

import rpy2.robjects as ro
from pydantic import Field
from pydantic.dataclasses import dataclass
from rpy2.rinterface_lib.embedded import RRuntimeError

from components.functional_analysis.base import FunctionalAnalysisBase
from components.functional_analysis.orgdb import OrgDB
from r_wrappers.cluster_profiler import enricher, gsea


class Config:
    arbitrary_types_allowed = True


@dataclass(config=Config)
class MSIGDBora(FunctionalAnalysisBase):
    """
    Over-representation analysis for Molecular Signatures Database (MSigDB)

    Args:
        func_kwargs: Additional arguments for functional analysis function.
    """

    term2gene: Any = None
    func_kwargs: dict = Field(default_factory=dict)

    def __post_init__(self):
        # 1. Get functional result
        self.func_result = enricher(
            self.filtered_genes.names,
            universe=self.background_genes.names,
            TERM2GENE=self.term2gene,
            **self.func_kwargs,
        )
        super().__post_init__()

    def ridgeplot(self, **kwargs):
        raise NotImplementedError("ORA result cannot be used for GSEA plots.")

    def gseaplot(self, gene_set_id: int, **kwargs):
        raise NotImplementedError("ORA result cannot be used for GSEA plots.")

    def plot_all_gsea(self, **kwargs):
        raise NotImplementedError("ORA result cannot be used for GSEA plots.")

    def plot_all(self, **kwargs):
        self.plot_all_ora(**kwargs)


@dataclass(config=Config)
class MSIGDBgsea(FunctionalAnalysisBase):
    """
    Gene-set Enrichment analysis for Molecular Signatures Database (MSigDB)

    Args:
        func_kwargs: Additional arguments for functional analysis function.
    """

    term2gene: Any = None
    func_kwargs: dict = Field(default_factory=dict)

    def __post_init__(self):
        # 1. Get functional result
        self.func_result = gsea(
            self.background_genes, TERM2GENE=self.term2gene, **self.func_kwargs
        )
        super().__post_init__()

    def plot_all(self, **kwargs):
        self.plot_all_gsea(**kwargs)


def run_msigdb_ora(
    background_genes: ro.FloatVector,
    org_db: OrgDB,
    filtered_genes: ro.FloatVector,
    files_prefix: Path,
    plots_prefix: Path,
    term2gene: Any,
):
    try:
        ora = MSIGDBora(
            background_genes,
            org_db,
            filtered_genes,
            files_prefix,
            plots_prefix,
            term2gene=term2gene,
        )
        ora.save_csv()
        ora.plot_all()
    except RRuntimeError as e:
        logging.warning(
            f"[{plots_prefix.name}] Error computing functional result: \n\t{e}"
        )


def run_msigdb_gsea(
    background_genes: ro.FloatVector,
    org_db: OrgDB,
    filtered_genes: ro.FloatVector,
    files_prefix: Path,
    plots_prefix: Path,
    term2gene: Any,
):
    try:
        gsea = MSIGDBgsea(
            background_genes,
            org_db,
            filtered_genes,
            files_prefix,
            plots_prefix,
            term2gene=term2gene,
        )
        gsea.save_csv()
        gsea.plot_all()
    except RRuntimeError as e:
        logging.warning(
            f"[{plots_prefix.name}] Error computing functional result: \n\t{e}"
        )
