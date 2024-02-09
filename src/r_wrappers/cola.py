"""
    Wrappers for R package cola

    All functions have pythonic inputs and outputs.

    Note that the arguments in python use "_" instead of ".".
    rpy2 does this transformation for us.
    Eg:
        R --> data.category
        Python --> data_category
"""

from pathlib import Path

import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from r_wrappers.utils import pd_df_to_rpy2_df

r_cola = importr("cola")


def run_all_consensus_partition_methods(
    data: pd.DataFrame,
    threads: int = 4,
    max_k: int = 5,
    save_dir: Path = None,
    **kwargs
):
    """
    Consensus partitioning for all combinations of methods.

    If a directory is provided, make HTML report from the ConsensusPartitionList object.

    Args:
        data: A data matrix of shape [n_features, n_samples]. Clustering is performed
            on samples.

    *ref docs:
        https://rdrr.io/bioc/cola/man/run_all_consensus_partition_methods.html
        https://rdrr.io/bioc/cola/man/cola_report-ConsensusPartitionList-method.html
    """
    cores = min(threads, max_k - 1)
    rl = r_cola.run_all_consensus_partition_methods(
        pd_df_to_rpy2_df(data), max_k=max_k, cores=cores, **kwargs
    )

    if save_dir is not None:
        ro.r("cola_report")(rl, output_dir=str(save_dir), cores=cores, **kwargs)

    return rl


def hierarchical_partition(
    data: pd.DataFrame,
    threads: int = 4,
    max_k: int = 5,
    save_dir: Path = None,
    **kwargs
):
    """
    Hierarchical partition.

    If a directory is provided, make HTML report from the ConsensusPartitionList object.

    Args:
        data: A data matrix of shape [n_features, n_samples]. Clustering is performed
            on samples.

    *ref docs:
        https://rdrr.io/bioc/cola/man/hierarchical_partition.html
    """
    cores = min(threads, max_k - 1)
    rl = r_cola.hierarchical_partition(
        pd_df_to_rpy2_df(data), max_k=max_k, cores=cores, **kwargs
    )

    if save_dir is not None:
        ro.r("cola_report")(rl, output_dir=str(save_dir), cores=cores, **kwargs)

    return rl
