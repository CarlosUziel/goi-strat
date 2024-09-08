"""
    Wrappers for R package bsseq

    All functions have pythonic inputs and outputs.

    Note that the arguments in python use "_" instead of ".".
    rpy2 does this transformation for us.
    Eg:
        R --> data.category
        Python --> data_category
"""

from pathlib import Path
from typing import Any, List

import rpy2.robjects as ro
from rpy2.robjects.packages import importr

r_source = importr("bsseq")


def read_bismark(cov_files: List[Path], **kwargs) -> Any:
    """
    Parsing output from the Bismark alignment suite.

    See: https://rdrr.io/bioc/bsseq/man/read.bismark.html

    Args:
        cov_files: List of coverage files, obtained from running
            'bismark_methylation_extractor'
    """
    return r_source.read_bismark(
        files=ro.StrVector(list(map(str, cov_files))), **kwargs
    )


def get_coverage(bsseq_obj: Any, **kwargs) -> Any:
    """
    Obtain coverage for BSseq objects.

    See: https://rdrr.io/bioc/bsseq/man/getCoverage.html

    Args:
        bsseq_obj: An object of class BSseq.
    """
    return r_source.getCoverage(bsseq_obj, **kwargs)
