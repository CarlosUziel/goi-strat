"""
Wrappers for R package pathview

All functions have pythonic inputs and outputs.

Note that the arguments in python use "_" instead of ".".
rpy2 does this transformation for us.
Eg:
    R --> ann_df.category
    Python --> data_category
"""

import os
import re
from pathlib import Path

import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

r_pathview = importr("pathview")


def pathview(
    gene_data: ro.FloatVector,
    pathway_id: str,
    pathway_name: str,
    save_dir: Path,
    **kwargs,
):
    """
    Pathview is a tool set for pathway based data integration and visualization. It
    maps and renders user data on relevant pathway graphs. All users need is to supply
    their gene or compound data and specify the target pathway. Pathview automatically
    downloads the pathway graph data, parses the data file, maps user data to the
    pathway, and render pathway graph with the mapped data. Pathview generates both
    native KEGG view and Graphviz views for pathways. keggview.native and
    keggview.graph are the two viewer functions, and pathview is the main function
    providing a unified interface to downloader, parser, mapper and viewer functions.

    *ref docs docs in https://rdrr.io/bioc/pathview/man/pathview.html
    """
    current_wd = os.getcwd()
    os.chdir(save_dir)
    pathway_name = re.sub("_{2,}", "_", re.sub(r"[\s\-\/]", "_", pathway_name.lower()))
    with localconverter(ro.default_converter):
        r_pathview.pathview(
            gene_data=gene_data,
            pathway_id=pathway_id,
            limit=ro.ListVector({"gene": max(gene_data, key=abs), "cpd": 1}),
            kegg_native=True,
            **kwargs,
        )
    os.chdir(current_wd)

    for f in save_dir.glob(f"{pathway_id}.*"):
        f.replace(str(f).replace(pathway_id, f"{pathway_id}_{pathway_name}"))
