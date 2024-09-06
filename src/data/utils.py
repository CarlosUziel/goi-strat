import contextlib
import os
import signal
import subprocess
from contextlib import contextmanager
from copy import deepcopy
from itertools import chain, combinations
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.power import TTestIndPower
from tqdm.rich import tqdm

from components.functional_analysis.orgdb import OrgDB
from r_wrappers.utils import map_gene_id


def gene_expression_levels(
    expr_df: pd.DataFrame,
    gene_expr_col: str,
    gene_expr_level: str,
    percentile: int = 10,
):
    """
    Given a column containing gene expression values for each sample,
    add an extra column with the gene name, where each sample is assigned a
    label depending on the relative expression level (i.e. "low", "mid" or
    "high").

    Args:
        expr_df: Contains a column with the log counts of a given gene.
        percentile: Percentile to include in both sides of the ranked gene
            Iterable (tail). If percentile is 10, then the
            first 10% and the last 10% of the ranked gene Iterable will belong
            to groups "low" and "high", respectively.
        gene_expr_col: Column containing expression values.
        gene_expr_level: Column name of the new column containing the
            expression levels.
    """
    # 0. Get user-provided percentiles
    expr_df = deepcopy(expr_df)
    p0, p1 = np.percentile(expr_df[gene_expr_col], (percentile, 100 - percentile))

    # 1. Classify data into three groups
    expr_df.loc[expr_df[gene_expr_col] < p0, gene_expr_level] = "low"
    expr_df.loc[
        (expr_df[gene_expr_col] >= p0) & (expr_df[gene_expr_col] <= p1), gene_expr_level
    ] = "mid"
    expr_df.loc[expr_df[gene_expr_col] > p1, gene_expr_level] = "high"

    return expr_df


def run_cmd(cmd: Iterable[str], log_path: Optional[Path] = None) -> Dict[str, str]:
    """
    Runs console command. Saves log if log_path is provided and is
        valid. Piping with '|' is allowed.
    Args:
        cmd: Iterable, where each element is a component of the command.
            E.g.: ['/bin/prog', '-i', 'data.txt', '-o', 'more data.txt']
        log_path: Optionally, add a path to store stdout and stderr.
    """
    # 0. Run commands separated by pipes
    process_output = None
    try:
        for i, process in enumerate(" ".join([str(x) for x in cmd]).split("|")):
            process_input = (
                process_output.stdout if process_output is not None else None
            )
            process_output = subprocess.run(
                process.strip().split(" "),
                input=process_input,
                check=True,
                capture_output=True,
                universal_newlines=True,
            )
    except subprocess.CalledProcessError as e:
        print(e)

    # 1. Save stdout and stderr if command execution was successful and a
    # file path is provided
    if process_output is not None:
        logs = {
            "stderr": str(process_output.stderr),
            "stdout": str(process_output.stdout),
        }
        if log_path is not None and log_path.suffix == ".log":
            log_path.write_text(
                f'stderr:\n {logs["stderr"]} \n\n stdout:\n {logs["stdout"]}'
            )

        return logs


def parallelize_star(
    func: Callable, inputs: Iterable, threads: int = 8, method: str = "spawn"
):
    with get_context(method).Pool(threads, maxtasksperchild=1) as pool:
        return pool.starmap(func, tqdm(inputs))


def parallelize_map(
    func: Callable, inputs: Iterable, threads: int = 8, method: str = "spawn"
):
    with get_context(method).Pool(threads, maxtasksperchild=1) as pool:
        return list(
            tqdm(
                pool.imap_unordered(func, inputs),
                total=len(inputs),
            )
        )


def filter_genes_wrt_annotation(
    genes: Iterable[str], org_db: OrgDB, from_type: str = "ENSEMBL"
) -> Iterable[str]:
    """
    Given a Iterable of genes, annotate and filter them based on those annotations
        according to some predefined criteria.

    Args:
        genes: Iterable of genes to be filtered.
        org_db: Organism annotation database.
        from_type: Gene IDs scheme of the gene Iterable.

    Returns:
        A filtered annotation data frame, with the same columns and index as
        the original where some rows have been removed.

    """
    # 1. Filter genes without ENTREZID
    genes_entrezid = map_gene_id(genes, org_db, from_type, "ENTREZID")
    genes_entrezid = (
        genes_entrezid[~genes_entrezid.str.contains("/", na=False)]
        .dropna()
        .drop_duplicates(keep=False)
    )

    # 2. Remove genes with unwanted characteristics
    gene_types = map_gene_id(genes_entrezid.index, org_db, from_type, "GENETYPE")
    gene_types = gene_types[~genes_entrezid.str.contains("/", na=False)].dropna()[
        gene_types.isin(["protein-coding", "ncRNA"])
    ]

    return gene_types.index.tolist()


def filter_df(
    df: pd.DataFrame, filter_values: Dict[str, Iterable[Any]]
) -> pd.DataFrame:
    """
        Filter pandas dataframe by matching targets for multiple columns.

    Args:
        df: pandas DataFrame object to be filtered
        filter_values: Dictionary of the form:
                `{<field>: <target_values_Iterable>}`
            used to filter columns data.
    """
    # ensure that all fields are valid
    assert all([k in df.columns for k in filter_values.keys()])

    # filter dataframe
    return df[
        np.logical_and.reduce(
            [
                df[column].isin(target_values)
                for column, target_values in filter_values.items()
            ]
        )
    ]


def select_data_classes(
    metadata: pd.DataFrame, classes_filters: Iterable[Dict[str, Iterable[Any]]]
) -> Iterable[Iterable[Any]]:
    """
        Given a pandas dataframe describing data, with size
            (samples x features), apply different filters to obtain the
            unique class IDs. Thus, the IDs of the different classes cannot
            overlap.

        The dataframe index is used as unique class IDs (should be the sample ID)

        Args:
            metadata: data metadata
            classes_filters: Iterable of dictionaries. Each dictionary of the form:
                `{<field>: <target_values_Iterable>}`, used to filter columns data.

    Returns:
        A Iterable of length equal to the number of classes, each element contains
            the unique sample IDs belonging to each class.
    """
    # 1. Filter dataframe and get sample IDs for each class
    class_samples_ids = [
        filter_df(metadata, classes_filters).index
        for classes_filters in classes_filters
    ]

    # 2. Check that the samples of the different classes do not intersect
    assert (
        len(set(metadata.index).intersection(*class_samples_ids)) == 0
    ), "There are overlapping samples among classes, please check the class filters"

    # 3. Return class ids
    return class_samples_ids


def ranges_overlap(ranges: Iterable[Tuple[float, float]]):
    """
    Find whether the passed ranges overlap. True is only return if ALL ranges overlap.
    For an overlap, there exists some number X which is in all ranges, i.e.

        A1 <= C <= B1 for all ranges [Ai:Bi]

    Asuming the ranges are well-formed (so that A <= B for all ranges) then it is
        sufficient to test:

        A1 <= B2 && B1 <= A2 (StartA <= EndB) and (EndA >= StartB) for each pair of
    ranges.

    Args:
        ranges: Iterable of ranges to check overlap for.
    """
    for (a_min, a_max), (b_min, b_max) in zip(ranges[:-1], ranges[1:]):
        if not ((a_min <= b_max) and (b_min <= a_max)):
            return False
    return True


def get_overlapping_features(
    data_df: pd.DataFrame, class_samples_ids: Iterable[Iterable[Any]]
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Get genes whose values do not overlap among classes.

    Args:
        data_df: Gene expression data of shape [n_samples, n_genes]
        class_samples_ids: Iterable of Iterables of sample IDs, one per class.
    """
    assert len(class_samples_ids) >= 2, "At least two sets of samples must be provided."
    data_df_class = deepcopy(data_df)

    # 1. Assign class IDs to samples
    for i, samples_ids in enumerate(class_samples_ids):
        data_df_class.loc[samples_ids, "class"] = i

    # 2. Get min and max values per class
    data_df_ranges = data_df_class.groupby("class").agg(("min", "max")).transpose()

    # 3. Compute gene expression ranges overlap among classes
    return (
        data_df_ranges.groupby(level=0).apply(
            lambda gene: ranges_overlap([x[1] for x in gene.items()])
        ),
        data_df_ranges,
    )


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    # https://docs.python.org/2/library/itertools.html#recipes
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def supress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)

    return wrapper


def calculate_power(effect_size: float, alpha: float, n1: int, n2: int) -> float:
    """
    Calculate the power of a two-sample t-test.

    Parameters:
    - effect_size: Cohen's d, the standardized difference between two means.
    - alpha: Significance level of the test.
    - n1: Sample size for group 1.
    - n2: Sample size for group 2.

    Returns:
    - float: The power of the test.
    """
    # Initialize the power analysis object
    power_analysis = TTestIndPower()

    # Calculate the power
    power = power_analysis.solve_power(
        effect_size=effect_size,
        nobs1=n1,
        alpha=alpha,
        ratio=n2 / n1,
        alternative="two-sided",
    )

    return power
