import logging
from pathlib import Path

import rpy2.robjects as ro
from pydantic import Field
from pydantic.dataclasses import dataclass
from rpy2.rinterface_lib.embedded import RRuntimeError

from components.functional_analysis.base import FunctionalAnalysisBase
from components.functional_analysis.orgdb import OrgDB
from r_wrappers.cluster_profiler import enrich_go, gse_go


class Config:
    arbitrary_types_allowed = True


@dataclass(config=Config)
class GOora(FunctionalAnalysisBase):
    """
    Over-representation analysis for Gene Ontology terms.

    Args:
        func_kwargs: Additional arguments for functional analysis function.
    """

    func_kwargs: dict = Field(default_factory=dict)

    def __post_init_post_parse__(self):
        # 1. Get functional result
        self.func_result = enrich_go(
            self.filtered_genes.names,
            universe=self.background_genes.names,
            org_db=self.org_db,
            **self.func_kwargs,
        )
        super().__post_init_post_parse__()

    def ridgeplot(self, **kwargs):
        raise NotImplementedError("ORA result cannot be used for GSEA plots.")

    def gseaplot(self, gene_set_id: int, **kwargs):
        raise NotImplementedError("ORA result cannot be used for GSEA plots.")

    def plot_all_gsea(self, **kwargs):
        raise NotImplementedError("ORA result cannot be used for GSEA plots.")

    def plot_all(self, **kwargs):
        self.goplot(**kwargs)
        self.plot_all_ora(**kwargs)


@dataclass(config=Config)
class GOgsea(FunctionalAnalysisBase):
    """
    Gene-set Enrichment analysis for Gene Ontology terms.

    Args:
        func_kwargs: Additional arguments for functional analysis function.
    """

    func_kwargs: dict = Field(default_factory=dict)

    def __post_init_post_parse__(self):
        # 1. Get functional result
        self.func_result = gse_go(
            self.background_genes, org_db=self.org_db, **self.func_kwargs
        )
        super().__post_init_post_parse__()

    def plot_all(self, **kwargs):
        self.plot_all_gsea(**kwargs)


def run_go_ora(
    background_genes: ro.FloatVector,
    org_db: OrgDB,
    filtered_genes: ro.FloatVector,
    files_prefix: Path,
    plots_prefix: Path,
    ont: str,
):
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
):
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
