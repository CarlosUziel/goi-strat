import logging
from typing import Any

import rpy2.robjects as ro
from rpy2.robjects.packages import importr

r_annotation_hub = importr("AnnotationHub")
double_brackets = ro.r("function(obj, idx){return(obj[[idx]])}")


class OrgDB:
    """
    OrgDB (Organism Database) class for accessing Bioconductor organism-specific annotation packages.

    This class provides access to organism-specific annotation databases through Bioconductor's
    AnnotationHub. It creates a connection to the appropriate database based on the species name.

    Attributes:
        species (str): The species name to use for querying the AnnotationHub.
            Default is "Homo sapiens".

    Examples:
        >>> from components.functional_analysis.orgdb import OrgDB
        >>> org_db = OrgDB(species="Homo sapiens")
        >>> # Access the database object
        >>> db = org_db.db
    """

    def __init__(self, species: str = "Homo sapiens") -> None:
        """
        Initialize the OrgDB instance with the specified species.

        Args:
            species: The species name to use for querying the AnnotationHub.
                Default is "Homo sapiens". Other common options include
                "Mus musculus", "Rattus norvegicus", etc.
        """
        self.species = species
        # self.db_lock = Manager().Lock()

    @property
    def db(self) -> Any:
        """
        Access the organism-specific annotation database via AnnotationHub.

        Queries AnnotationHub for the appropriate organism package and returns
        the database object. First tries to connect to the remote hub, and if that
        fails, attempts to use a local hub.

        Returns:
            An R object representing the organism database.

        Raises:
            Exception: If there's an error connecting to AnnotationHub or retrieving
                the database, the error is logged and may be re-raised.
        """
        try:
            anno_hub = ro.r("function(){suppressMessages(AnnotationHub())}")()
        except Exception as e:
            logging.warning(e)
            anno_hub = ro.r(
                "function(){suppressMessages(AnnotationHub(localHub=TRUE))}"
            )()

        return double_brackets(
            anno_hub,
            r_annotation_hub.query(
                anno_hub, ro.StrVector((self.species, "^org.*"))
            ).names[0],
        )
