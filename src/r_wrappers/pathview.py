"""
Wrappers for R package pathview

All functions have pythonic inputs and outputs.

Note that the arguments in python use "_" instead of ".".
rpy2 does this transformation for us.

Example:
R --> data.category
Python --> data_category
"""

import os
import re
from pathlib import Path
from typing import Any

import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

r_pathview = importr("pathview")


def pathview(
    gene_data: ro.FloatVector,
    pathway_id: str,
    pathway_name: str,
    save_dir: Path,
    **kwargs: Any,
) -> None:
    """Visualize gene expression data on KEGG pathway graphs.

    Pathview maps and renders user data on relevant pathway graphs from the
    KEGG database. It automatically downloads the pathway graph data, parses
    the data file, maps the provided gene data to the pathway, and renders
    a pathway graph with the mapped data overlaid on it.

    Args:
        gene_data: A named vector of gene expression values (typically log fold changes
            or other statistics). The names should be gene IDs that can be mapped to
            the KEGG pathway (usually Entrez gene IDs).
        pathway_id: The KEGG pathway ID (e.g., "hsa04110" for human cell cycle pathway).
        pathway_name: A descriptive name for the pathway, which will be used in the
            output file name.
        save_dir: Directory where the generated pathway images will be saved.
        **kwargs: Additional arguments to pass to the pathview function.
            Common parameters include:
            - species: KEGG species code (default: determined from pathway_id).
            - gene_idtype: Type of gene IDs provided (default: "entrez").
            - out_suffix: Suffix to add to output file names.
            - kegg_native: Whether to generate native KEGG view (default: TRUE).
            - same_layer: Whether to draw all genes on the same layer (default: FALSE).
            - map_color: Color scales for gene data.
            - low: Lower bound of gene data for color mapping.
            - mid: Midpoint of gene data for color mapping.
            - high: Upper bound of gene data for color mapping.

    Returns:
        None: The function saves pathway visualization files to the specified directory
        but doesn't return any value.

    Notes:
        This function changes the working directory to save_dir temporarily while
        executing the pathview command, then changes back to the original directory.
        It also renames the output files to include the pathway name in addition
        to the pathway ID.

    References:
        https://rdrr.io/bioc/pathview/man/pathview.html
        https://doi.org/10.1093/bioinformatics/btt285
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

    # Rename output files to include the pathway name for better identification
    for f in save_dir.glob(f"{pathway_id}.*"):
        f.replace(str(f).replace(pathway_id, f"{pathway_id}_{pathway_name}"))
