"""
Wrappers for R package rbioapi

All functions have pythonic inputs and outputs.

Note that the arguments in python use "_" instead of ".".
rpy2 does this transformation for us.

Example:
    R --> data.category
    Python --> data_category
"""

import logging
from multiprocessing import Lock
from multiprocessing.synchronize import Lock as LockBase
from pathlib import Path
from time import sleep
from typing import Any, Iterable

import pandas as pd
import rpy2.robjects as ro
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects.packages import importr

from r_wrappers.utils import rpy2_df_to_pd_df

r_rbioapi = importr("rbioapi")
r_ggplot = importr("ggplot2")


def string_map_ids(
    symbol_genes: Iterable[str],
    species: int = 9606,
    api_lock: LockBase = Lock(),
    **kwargs: Any,
) -> pd.DataFrame:
    """Calls STRING's API to convert a set of identifiers to STRING identifiers.

    Although you can call STRING services with a variety of common identifiers,
    it is recommended by STRING's documentation that you first map your
    protein/gene IDs to STRING IDs and then proceed with other STRING functions.

    Args:
        symbol_genes: List of genes to map.
        species: NCBI Taxonomy identifier; Human Taxonomy ID is 9606. (Recommended,
            but optional if your input is less than 100 IDs).
        api_lock: Multiprocessing lock to access STRINGDB API.
        **kwargs: Additional arguments to pass to the R function rba_string_map_ids.

    Returns:
        DataFrame with mapped STRING identifiers.

    Raises:
        RRuntimeError: If there is an error in the R function call. Will retry after 300 seconds.

    Example:
        >>> string_map_ids(['BRCA1', 'TP53'], species=9606)
    """
    while True:
        try:
            with api_lock:
                return rpy2_df_to_pd_df(
                    r_rbioapi.rba_string_map_ids(
                        ro.StrVector(symbol_genes),
                        species=ro.r("as.numeric")(species),
                        **kwargs,
                    )
                )
        except RRuntimeError as e:
            logging.error(e)
            sleep(300)


def string_network_image(
    proteins_ids: Iterable[str],
    image_format: str,
    save_path: Path,
    species: int = 9606,
    api_lock: LockBase = Lock(),
    **kwargs: Any,
) -> None:
    """Produces a static image of the interaction networks among input proteins.

    Depending on whether you supplied a single protein ID or multiple protein IDs,
    this function will produce a static image of the interaction networks among your
    input proteins or/and with other proteins. See the "Arguments" section to learn
    more about how you can modify the network image.

    Args:
        proteins_ids: A list of protein IDs.
        image_format: Image format code. Can be "png" or "svg".
        save_path: Path to save the image to.
        species: NCBI Taxonomy identifier; Human Taxonomy ID is 9606. (Recommended,
            but optional if your input is less than 100 IDs).
        api_lock: Multiprocessing lock to access STRINGDB API.
        **kwargs: Additional arguments to pass to the R function rba_string_network_image.

    Raises:
        AssertionError: If image_format is not "png" or "svg", or save_path suffix doesn't match format.
        RRuntimeError: If there is an error in the R function call. Will retry after 300 seconds.

    Example:
        >>> string_network_image(['ENSP00000269305', 'ENSP00000324856'],
        ...                     'png', Path('/path/to/output.png'))
    """
    assert image_format in ["png", "svg"]
    assert save_path.suffix == f".{image_format}"

    while True:
        try:
            with api_lock:
                r_rbioapi.rba_string_network_image(
                    ro.StrVector(proteins_ids),
                    image_format=("highres_image" if image_format == "png" else "svg"),
                    save_image=str(save_path),
                    species=ro.r("as.numeric")(species),
                    **kwargs,
                )
                return
        except RRuntimeError as e:
            logging.error(e)
            sleep(300)


def string_interactions_network(
    proteins_ids: Iterable[str],
    species: int = 9606,
    required_score: int = 500,
    network_type: str = "functional",
    api_lock: LockBase = Lock(),
    **kwargs: Any,
) -> pd.DataFrame:
    """Retrieves STRING interaction pairs among input protein IDs.

    This function retrieves STRING interaction pairs among your input protein IDs,
    with the combined score and separate score for each STRING score channel. You can
    further expand your network to a defined size by providing "add_node" parameter.

    Args:
        proteins_ids: A list of protein IDs.
        species: NCBI Taxonomy identifier; Human Taxonomy ID is 9606. (Recommended,
            but optional if your input is less than 100 IDs).
        required_score: A minimum interaction score for an interaction to be
            included. If not supplied, the threshold will be applied by STRING based
            on the network:

            - Low Confidence = 150
            - Medium Confidence = 400
            - High Confidence = 700
            - Highest confidence = 900

        network_type: Should be one of:

            - "functional": The edges indicate both physical and functional
              associations.
            - "physical": The edges indicate that two proteins have a physical
              interaction or are parts of a complex.

        api_lock: Multiprocessing lock to access STRINGDB API.
        **kwargs: Additional arguments to pass to the R function rba_string_interactions_network.

    Returns:
        DataFrame with interaction pairs and scores.

    Raises:
        RRuntimeError: If there is an error in the R function call. Will retry after 300 seconds.

    Example:
        >>> string_interactions_network(['ENSP00000269305', 'ENSP00000324856'],
        ...                           required_score=700)
    """
    while True:
        try:
            with api_lock:
                return rpy2_df_to_pd_df(
                    r_rbioapi.rba_string_interactions_network(
                        ro.StrVector(proteins_ids),
                        species=ro.r("as.numeric")(species),
                        required_score=ro.r("as.numeric")(required_score),
                        network_type=network_type,
                        **kwargs,
                    )
                )
        except RRuntimeError as e:
            logging.error(e)
            sleep(300)
