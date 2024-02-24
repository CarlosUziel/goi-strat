"""
    Wrappers for R package SBGNview

    All functions have pythonic inputs and outputs.

    Note that the arguments in python use "_" instead of ".".
    rpy2 does this transformation for us.
    Eg:
        R --> ann_df.category
        Python --> data_category
"""

from pathlib import Path

import rpy2.robjects as ro
from rpy2.robjects.packages import importr

r_sbgnview = importr("SBGNview")


def sbgn_view(
    gene_data: ro.FloatVector,
    pathway_id: str,
    pathway_name: str,
    save_dir: Path,
    **kwargs
):
    """
    This is the main function to map, integrate and render omics data on pathway
    graphs. Two inputs are needed: 1. A pathway file in SBGN-ML format and 2.
    gene and/or compound omics data. The function generates image file of a
    pathway graph with the omics data mapped to the glyphs and rendered as
    pseudo-colors. If no gene and/or compound omics data is supplied to the
    function, the function will output the SVG image file (and other selected
    file formats) of the parsed input file . This is useful for viewing the
    pathway graph without overlaid omics data. This function is similar to
    Pathview except the pathways are rendered with SBGN notation. In addition,
    users can control more graph properties including node/edge attributes. We
    collected SBGN-ML files from several pathway databases: Reactome, MetaCyc,
    MetaCrop, PANTHER and SMPDB. Given a vector of patway IDs, SBGNview can
    automatically download and use these SBGN-ML files. To map omics data to
    glyphs, user just needs to specify omics data ID types. When using user
    customized SBGN-ML files, users need to provide a mapping file from omics
    data's molecule IDs to SBGN-ML file's glyph IDs.

    *ref docs docs in https://rdrr.io/bioc/SBGNview/man/SBGNview.html
    """
    pathway_name = pathway_name.lower().replace(" ", "_").replace("-", "_")
    r_sbgnview.SBGNview(
        gene_data=gene_data,
        input_sbgn=pathway_id,
        sbgn_dir=str(save_dir),
        output_file=str(save_dir.joinpath(pathway_name)),
        output_formats=ro.StrVector(["svg", "pdf"]),
        **kwargs
    )
