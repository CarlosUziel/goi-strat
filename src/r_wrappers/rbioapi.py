"""
Wrappers for R package rbioapi

All functions have pythonic inputs and outputs.

Note that the arguments in python use "_" instead of ".".
rpy2 does this transformation for us.
Eg:
    R --> data.category
    Python --> data_category
"""

import logging
from multiprocessing import Lock
from pathlib import Path
from time import sleep
from typing import Iterable

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
    api_lock: Lock = Lock(),
    **kwargs,  # type: ignore
) -> pd.DataFrame:
    """This function Calls STRING's API to Convert a set of identifiers to STRING
    Identifiers. Although You can call STRING services with a variety of common
    identifiers, It is recommended by STRING's documentations that you first map Your
    Protein/genes IDs to STRING IDs and then proceed with other STRING's functions.

    See:  https://rbioapi.moosa-r.com/reference/rba_string_map_ids.html

    Args:
        symbol_genes: List of genes to map.
        species: NCBI Taxonomy identifier; Human Taxonomy ID is 9606. (Recommended,
            but optional if your input is less than 100 IDs).
        api_lock: Multiprocessing lock to access STRINGDB API.
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
    api_lock: Lock = Lock(),  # type: ignore
    **kwargs,
) -> None:
    """Depending on that you supplied a single protein ID or more than one protein ID,
    this function will produce a static image of the interaction networks among your
    input proteins or/and with other proteins. See the "Arguments" section to learn
    more about how you can modify the network image.

    See: https://rbioapi.moosa-r.com/reference/rba_string_network_image.html

    Args:
        proteins_ids: A list of protein IDs.
        image_format: Image format code. Can be "png" or "svg".
        save_path: Path to save the image to.
        species: NCBI Taxonomy identifier; Human Taxonomy ID is 9606. (Recommended,
            but optional if your input is less than 100 IDs).
        api_lock: Multiprocessing lock to access STRINGDB API.

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
    api_lock: Lock = Lock(),  # type: ignore
    **kwargs,
) -> pd.DataFrame:
    """This function will retrieve Sting interaction pairs among your input protein ids,
    with the combined score and separate score for each STRING score channels. You can
    further expand your network to a defined size by providing "add_node" parameter.

    See: https://rdrr.io/cran/rbioapi/man/rba_string_interactions_network.html

    Args:
        proteins_ids: A list of protein IDs.
        species: NCBI Taxonomy identifier; Human Taxonomy ID is 9606. (Recommended,
            but optional if your input is less than 100 IDs).
        required_score: A minimum of interaction score for an interaction to be
            included. if not supplied, the threshold will be applied by STRING Based
            in the network.
            - low Confidence = 150
            - Medium Confidence = 400
            - High Confidence = 700
            - Highest confidence = 900
        network_type: Should be one of:
            - "functional": The edge's indicate both physical and functional
                associations.
            - "physical": The edges indicate that two proteins have a physical
                interaction or are parts of a complex.
        api_lock: Multiprocessing lock to access STRINGDB API.
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
