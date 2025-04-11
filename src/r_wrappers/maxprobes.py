"""
Wrappers for R package maxprobes

All functions have pythonic inputs and outputs.

Note that the arguments in python use "_" instead of ".".
rpy2 does this transformation for us.

Example:
R --> data.category
Python --> data_category
"""

from typing import Any

from rpy2.robjects.packages import importr

r_maxprobes = importr("maxprobes")


def xreactive_probes(array_type: str):
    """
    Get information about cross-reactive probes.

    Reference documentation:
        https://rdrr.io/github/markgene/maxprobes/man/xreactive_probes.html

    Args:
        array_type: A character scalar. One of "EPIC", "450K". Defaults to
          "EPIC".
    """
    return r_maxprobes.xreactive_probes(array_type)


def drop_xreactive_loci(minfi_obj: Any):
    """
    Remove cross-reactive probes from a minfi object.

    Reference documentation:
        https://rdrr.io/github/markgene/maxprobes/man/dropXreactiveLoci.html

    Args:
        minfi_obj: A minfi object

    """
    return r_maxprobes.dropXreactiveLoci(minfi_obj)
