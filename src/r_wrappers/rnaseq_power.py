"""
    Wrappers for R package RNASeqPower

    All functions have pythonic inputs and outputs.

    Note that the arguments in python use "_" instead of ".".
    rpy2 does this transformation for us.
    Eg:
        R --> data.category
        Python --> data_category
"""

from rpy2.robjects.packages import importr

r_rnaseq_power = importr("RNASeqPower")


def rnapower(**kwargs):
    """
    Sample size and power computation for RNA-seq studies.

    *ref docs: https://rdrr.io/bioc/RNASeqPower/man/rnapower.html
    """
    return r_rnaseq_power.rnapower(**kwargs)
