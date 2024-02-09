import logging
from pathlib import Path

import rpy2.robjects as ro
from pydantic import Field
from pydantic.dataclasses import dataclass
from rpy2.rinterface_lib.embedded import RRuntimeError

from components.functional_analysis.base import FunctionalAnalysisBase
from components.functional_analysis.orgdb import OrgDB
from r_wrappers.cluster_profiler import enrich_kegg, gse_kegg


class Config:
    arbitrary_types_allowed = True


@dataclass(config=Config)
class KEGGora(FunctionalAnalysisBase):
    """
    Over-representation analysis for KEGG Pathways.

    Args:
        func_kwargs: Additional arguments for functional analysis function.
    """

    func_kwargs: dict = Field(default_factory=dict)

    def __post_init_post_parse__(self):
        # 1. Get functional result
        self.func_result = enrich_kegg(
            self.filtered_genes.names,
            universe=self.background_genes.names,
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
        self.plot_all_ora(**kwargs)
        self.pathview(gene_data=self.filtered_genes, **kwargs)


@dataclass(config=Config)
class KEGGgsea(FunctionalAnalysisBase):
    """
    Gene-set Enrichment analysis for KEGG Pathways.

    Args:
        func_kwargs: Additional arguments for functional analysis function.
    """

    func_kwargs: dict = Field(default_factory=dict)

    def __post_init_post_parse__(self):
        # 1. Get functional result
        self.func_result = gse_kegg(self.background_genes, **self.func_kwargs)
        super().__post_init_post_parse__()

    def plot_all(self, **kwargs):
        self.plot_all_gsea(**kwargs)
        self.pathview(gene_data=self.background_genes, **kwargs)


def run_kegg_ora(
    background_genes: ro.FloatVector,
    org_db: OrgDB,
    filtered_genes: ro.FloatVector,
    files_prefix: Path,
    plots_prefix: Path,
):
    try:
        ora = KEGGora(
            background_genes, org_db, filtered_genes, files_prefix, plots_prefix
        )
        ora.save_csv()
        ora.plot_all()
    except RRuntimeError as e:
        logging.warning(
            f"[{plots_prefix.name}] Error computing functional result: \n\t{e}"
        )


def run_kegg_gsea(
    background_genes: ro.FloatVector,
    org_db: OrgDB,
    filtered_genes: ro.FloatVector,
    files_prefix: Path,
    plots_prefix: Path,
):
    try:
        gsea = KEGGgsea(
            background_genes, org_db, filtered_genes, files_prefix, plots_prefix
        )
        gsea.save_csv()
        gsea.plot_all()
    except RRuntimeError as e:
        logging.warning(
            f"[{plots_prefix.name}] Error computing functional result: \n\t{e}"
        )
