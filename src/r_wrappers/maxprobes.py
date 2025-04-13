"""
Wrappers for R package maxprobes

All functions have pythonic inputs and outputs.

Note that the arguments in python use "_" instead of ".".
rpy2 does this transformation for us.

Example:
R --> data.category
Python --> data_category
"""

from typing import Any, List

from rpy2.robjects.packages import importr

r_maxprobes = importr("maxprobes")


def xreactive_probes(array_type: str = "EPIC") -> List[str]:
    """Get information about cross-reactive probes in methylation arrays.

    This function retrieves a list of cross-reactive probes for the specified
    methylation array platform. Cross-reactive probes are probes that can
    hybridize to multiple genomic locations, potentially leading to false results.

    Args:
        array_type: The type of methylation array. Must be one of "EPIC" or "450K".
            Default is "EPIC".

    Returns:
        List[str]: A list of probe IDs identified as cross-reactive.

    References:
        https://rdrr.io/github/markgene/maxprobes/man/xreactive_probes.html
    """
    return r_maxprobes.xreactive_probes(array_type)


def drop_xreactive_loci(minfi_obj: Any) -> Any:
    """Remove cross-reactive probes from a minfi object.

    This function filters out cross-reactive probes from a minfi methylation data
    object. These probes can map to multiple genomic locations and may lead to
    false methylation signals.

    Args:
        minfi_obj: A minfi object containing methylation data. This can be various
            types of minfi objects such as RGChannelSet, MethylSet, GenomicRatioSet, etc.

    Returns:
        Any: The input minfi object with cross-reactive probes removed.
            The return type matches the input type.

    References:
        https://rdrr.io/github/markgene/maxprobes/man/dropXreactiveLoci.html
    """
    return r_maxprobes.dropXreactiveLoci(minfi_obj)
