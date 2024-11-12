import logging
from itertools import chain
from pathlib import Path
from typing import Optional

import rpy2.robjects as ro
from pydantic import Field
from pydantic.dataclasses import dataclass
from rpy2.rinterface_lib.embedded import RRuntimeError

from components.functional_analysis.orgdb import OrgDB
from r_wrappers.gprofiler2 import gost, gost_plot, publish_gost_plot, publish_gost_table
from r_wrappers.utils import map_gene_id, rpy2_df_to_pd_df_manual, save_rds


class Config:
    arbitrary_types_allowed = True


@dataclass(config=Config)
class GPROFILER2base:
    """
    Base class for enrichment analysis of multiple gene sets using gprofiler2

    Args:
        background_genes: All genes considered for the experiment.
        filtered_genes: Differentially expressed genes, filtered by user criteria.
        files_prefix: Path to be prefixed to all generated files.
        plots_prefix: Path to be prefixed to all generated files.
        func_kwargs: Additional arguments for functional analysis function.
    """

    background_genes: ro.FloatVector
    org_db: OrgDB
    filtered_genes: Optional[ro.FloatVector] = None
    files_prefix: Path = Path("")
    plots_prefix: Path = Path("")
    func_kwargs: dict = Field(default_factory=dict)

    def __post_init__(self):
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

    def plot(self, **kwargs):
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

    def tableplot(self, **kwargs):
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

    def gostplot(self, **kwargs):
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

    def plot_all(self, **kwargs):
        self.plot(**kwargs)
        self.tableplot(**kwargs)
        self.gostplot(**kwargs)


@dataclass(config=Config)
class GPROFILER2ora(GPROFILER2base):
    """
    Over-representation analysis for multiple gene sets using gprofiler2

    Args:
        func_kwargs: Additional arguments for functional analysis function.
    """

    func_kwargs: dict = Field(default_factory=dict)

    def __post_init__(self):
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
    (pseudo) Gene-set enrichment analysis for multiple gene sets using gprofiler2

    Args:
        func_kwargs: Additional arguments for functional analysis function.
    """

    func_kwargs: dict = Field(default_factory=dict)

    def __post_init__(self):
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
):
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
):
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
