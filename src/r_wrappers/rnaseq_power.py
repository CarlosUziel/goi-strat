"""
Wrappers for R package RnaSeqSampleSize

All functions have pythonic inputs and outputs.

Note that the arguments in python use "_" instead of ".".
rpy2 does this transformation for us.

Example:
R --> data.category
Python --> data_category
"""

from typing import Any, Dict, Union

from rpy2.robjects.packages import importr

r_rnaseq_power = importr("RNASeqPower")


def rnapower(**kwargs: Dict[str, Any]) -> Union[float, Any]:
    """Sample size and power computation for RNA-seq studies.

    This function provides a wrapper for the RNASeqPower::rnapower function
    to calculate sample size or power for RNA-seq experiments.

    Args:
        **kwargs: Keyword arguments passed to the underlying R function.
            Common parameters include:
            - depth: Sequencing depth (per sample).
            - effect: The effect size (biological difference between conditions).
            - cv: Coefficient of variation per gene.
            - alpha: Significance level.
            - power: Power value. If not specified, will be calculated.
            - n: Number of samples. If not specified, will be calculated.

    Returns:
        Union[float, Any]: Either the power or sample size, depending on which parameter
        was not specified in the input, or other R objects if returned by the underlying function.

    References:
        https://rdrr.io/bioc/RNASeqPower/man/rnapower.html
    """
    return r_rnaseq_power.rnapower(**kwargs)
