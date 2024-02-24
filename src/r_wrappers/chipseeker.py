"""
    Wrappers for R package ChIPseeker

    All functions have pythonic inputs and outputs.

    Note that the arguments in python use "_" instead of ".".
    rpy2 does this transformation for us.
    Eg:
        R --> data.category
        Python --> data_category
"""

from pathlib import Path
from typing import Any

from rpy2.robjects.packages import importr

r_source = importr("ChIPseeker")


def read_peak_file(peak_file: Path, **kwargs) -> Any:
    """
    Read peak file and store in data.frame or GRanges object.

    *ref docs: https://rdrr.io/bioc/ChIPseeker/man/readPeakFile.html

    Args:
        peak_file: Mapped peaks file.
    """
    return r_source.readPeakFile(str(peak_file), **kwargs)
