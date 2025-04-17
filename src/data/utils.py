"""Utility functions for data processing, manipulation, and analysis.

This module provides utilities for genomic data analysis, including gene expression
level categorization, process management, parallel computation, and statistical operations.
"""

import contextlib
import os
import signal
import subprocess
from contextlib import contextmanager
from copy import deepcopy
from itertools import chain, combinations
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar

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
) -> pd.DataFrame:
    """Categorize gene expression values into low, mid, and high levels.

    Adds a new column to the dataframe that categorizes gene expression values
    into "low", "mid", or "high" based on specified percentile thresholds.

    Args:
        expr_df: DataFrame containing gene expression values
        gene_expr_col: Column name containing the gene expression values
        gene_expr_level: Name for the new column that will contain expression levels
        percentile: Percentile threshold for low/high classification (e.g., if 10,
            the bottom 10% will be "low" and top 10% will be "high")

    Returns:
        pd.DataFrame: Copy of input DataFrame with additional column for expression levels

    Note:
        Values below the lower percentile are labeled "low", values above the upper
        percentile are labeled "high", and values in between are labeled "mid".
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
    """Execute a shell command and capture its output.

    Runs a command in a subprocess and optionally saves the output to a log file.
    Supports piping multiple commands with '|'.

    Args:
        cmd: Sequence of command components to execute
            (e.g., ['ls', '-l', '/home'])
        log_path: Optional path to save stdout and stderr output

    Returns:
        Dict[str, str]: Dictionary with 'stdout' and 'stderr' keys containing
            the command's output

    Raises:
        subprocess.CalledProcessError: If the command returns a non-zero exit code

    Note:
        For piping, the command string is split on '|' and each part is executed
        as a separate process, with stdout from one piped to stdin of the next.
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
                f"stderr:\n {logs['stderr']} \n\n stdout:\n {logs['stdout']}"
            )

        return logs


T = TypeVar("T")
R = TypeVar("R")


def parallelize_star(
    func: Callable[..., R],
    inputs: Iterable[Tuple],
    processes: int = 8,
    method: str = "spawn",
) -> List[R]:
    """Execute a function on multiple inputs in parallel using starmap.

    Runs a function on multiple sets of arguments in parallel using a process pool
    with progress tracking via tqdm.

    Args:
        func: Function to execute in parallel
        inputs: Iterable of argument tuples to pass to the function
        processes: Number of parallel processes to use, defaults to 8
        method: Multiprocessing start method ('spawn', 'fork', or 'forkserver')

    Returns:
        List[R]: List of function results in the same order as inputs

    Note:
        Uses starmap which unpacks each input tuple as *args to the function.
        The 'spawn' method is more stable but slower than 'fork'.
    """
    with get_context(method).Pool(processes, maxtasksperchild=1) as pool:
        return pool.starmap(func, tqdm(inputs))


def parallelize_map(
    func: Callable[[T], R],
    inputs: Iterable[T],
    processes: int = 8,
    method: str = "spawn",
) -> List[R]:
    """Execute a function on multiple inputs in parallel using imap_unordered.

    Runs a function on multiple single arguments in parallel using a process pool
    with progress tracking via tqdm.

    Args:
        func: Function to execute in parallel (taking a single argument)
        inputs: Iterable of arguments to pass to the function
        processes: Number of parallel processes to use, defaults to 8
        method: Multiprocessing start method ('spawn', 'fork', or 'forkserver')

    Returns:
        List[R]: List of function results in potentially different order from inputs

    Note:
        Uses imap_unordered which may return results in a different order than inputs.
        This can be faster than ordered processing when execution times vary.
    """
    with get_context(method).Pool(processes, maxtasksperchild=1) as pool:
        return list(
            tqdm(
                pool.imap_unordered(func, inputs),
                total=len(inputs),
            )
        )


def filter_genes_wrt_annotation(
    genes: Iterable[str], org_db: OrgDB, from_type: str = "ENSEMBL"
) -> List[str]:
    """Filter genes based on annotation criteria from an organism database.

    Filters a list of genes by removing those that:
    1. Cannot be mapped to ENTREZID
    2. Have ambiguous mappings (containing "/")
    3. Are not protein-coding or ncRNA gene types

    Args:
        genes: List of gene identifiers to filter
        org_db: Organism annotation database object
        from_type: Source identifier type of the genes list (e.g., "ENSEMBL")

    Returns:
        List[str]: Filtered list of gene identifiers meeting the criteria

    Note:
        This function keeps only genes that can be uniquely mapped to ENTREZIDs
        and are either protein-coding or ncRNA gene types.
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
    """Filter DataFrame rows based on values in specified columns.

    Filters rows of a DataFrame by matching column values against specified targets.

    Args:
        df: DataFrame to be filtered
        filter_values: Dictionary mapping column names to allowable values,
            where only rows with matching values are kept

    Returns:
        pd.DataFrame: Filtered DataFrame containing only rows that match all criteria

    Raises:
        AssertionError: If any key in filter_values is not a column in the DataFrame

    Example:
        >>> filter_df(df, {'status': ['active', 'pending'], 'type': ['user']})
        # Returns rows where status is either 'active' or 'pending' AND type is 'user'
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
) -> List[Iterable[Any]]:
    """Extract non-overlapping sample IDs for different classes from metadata.

    Applies different filtering criteria to a metadata DataFrame to extract
    sample IDs for distinct classes, ensuring that no sample belongs to
    multiple classes.

    Args:
        metadata: DataFrame containing sample metadata with samples as index
        classes_filters: List of filter dictionaries, each defining a class
            by specifying column values to filter on

    Returns:
        List[Iterable[Any]]: List of sample ID iterables, one for each class

    Raises:
        AssertionError: If sample IDs overlap between different classes

    Example:
        >>> select_data_classes(metadata, [
        ...     {'condition': ['treated']},
        ...     {'condition': ['control']}
        ... ])
        # Returns [treated_sample_ids, control_sample_ids]
    """
    # 1. Filter dataframe and get sample IDs for each class
    class_samples_ids = [
        filter_df(metadata, classes_filter).index for classes_filter in classes_filters
    ]

    # 2. Check that the samples of the different classes do not intersect
    assert len(set(metadata.index).intersection(*class_samples_ids)) == 0, (
        "There are overlapping samples among classes, please check the class filters"
    )

    # 3. Return class ids
    return class_samples_ids


def ranges_overlap(
    range_1: Tuple[int, int], range_2: Tuple[int, int], dist: int = 0
) -> bool:
    """Check if two numeric ranges overlap or are within a specified distance.

    Determines if two ranges (represented as [start, end] tuples) overlap or
    are at most 'dist' units apart from each other.

    Args:
        range_1: First range as (start, end) tuple with 0-based coordinates
        range_2: Second range as (start, end) tuple with 0-based coordinates
        dist: Minimum distance between ranges to be considered proximal,
            0 means ranges must overlap

    Returns:
        bool: True if ranges overlap or are within specified distance, False otherwise

    Example:
        >>> ranges_overlap((10, 20), (15, 25))  # Returns True (overlapping)
        >>> ranges_overlap((10, 20), (21, 30))  # Returns False (not overlapping)
        >>> ranges_overlap((10, 20), (22, 30), dist=2)  # Returns True (within distance)
    """


def get_overlapping_features(
    data_df: pd.DataFrame, class_samples_ids: Iterable[Iterable[Any]]
) -> Tuple[pd.Series, pd.DataFrame]:
    """Identify features (genes/probes) whose value ranges don't overlap between classes.

    For each feature in the data, determines whether the range of values
    (min to max) overlaps between different sample classes.

    Args:
        data_df: Data matrix with samples as rows and features (genes/probes) as columns
        class_samples_ids: List of sample ID iterables, one iterable per class

    Returns:
        Tuple[pd.Series, pd.DataFrame]: A tuple containing:
            - Series of booleans indicating whether each feature's ranges overlap (True)
              or not (False) between classes
            - DataFrame containing the min/max ranges of each feature by class

    Raises:
        AssertionError: If fewer than two sets of sample IDs are provided

    Note:
        Features that don't overlap between classes might be good candidates for
        biomarkers or distinguishing characteristics.
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
    """Exception raised when a code block execution exceeds its time limit.

    This exception is used with the time_limit context manager to gracefully
    handle timeouts in operations that might take too long to complete.
    """

    pass


@contextmanager
def time_limit(seconds: int):
    """Context manager to limit execution time of a code block.

    Creates a context within which code must complete within a specified
    time limit, or a TimeoutException will be raised.

    Args:
        seconds: Maximum number of seconds the enclosed code block is allowed to run

    Yields:
        None: This context manager doesn't yield a value

    Raises:
        TimeoutException: If the code within the context doesn't complete within
            the specified time limit

    Example:
        >>> try:
        ...     with time_limit(5):
        ...         # Code that might take too long
        ...         time.sleep(10)
        ... except TimeoutException:
        ...     print("Operation timed out")
    """

    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def powerset(iterable: Iterable[T]) -> Iterable[Tuple[T, ...]]:
    """Generate the power set of an iterable.

    Creates all possible combinations of elements from an iterable, including
    the empty set and the full set.

    Args:
        iterable: Input sequence for which to generate the power set

    Returns:
        Iterable of tuples containing all possible combinations of the input elements

    Example:
        >>> list(powerset([1, 2, 3]))
        [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]

    Note:
        For an input set of n elements, the power set contains 2^n elements.
        Based on the itertools recipes from Python's documentation.
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def supress_stdout(func: Callable[..., R]) -> Callable[..., R]:
    """Decorator to suppress standard output from a function.

    Wraps a function to redirect its stdout to /dev/null, effectively
    silencing any print statements or other output written to stdout.

    Args:
        func: The function whose stdout should be suppressed

    Returns:
        Callable: Wrapped function that executes silently

    Example:
        >>> @supress_stdout
        ... def noisy_function():
        ...     print("This will not be displayed")
        >>> noisy_function()
        # No output will be shown
    """

    def wrapper(*a, **ka):
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)

    return wrapper


def calculate_power(effect_size: float, alpha: float, n1: int, n2: int) -> float:
    """Calculate the power of a two-sample t-test.

    Determines the probability of detecting an effect of a given size when
    comparing two samples of specified sizes.

    Args:
        effect_size: Cohen's d, the standardized difference between two means
        alpha: Significance level of the test (e.g., 0.05)
        n1: Sample size for group 1
        n2: Sample size for group 2

    Returns:
        float: The statistical power of the test (0 to 1)

    Note:
        Power values closer to 1 indicate a higher probability of detecting
        a true effect when it exists.
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
