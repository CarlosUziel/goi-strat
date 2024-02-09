import logging

import rpy2.robjects as ro
from rpy2.robjects.packages import importr

r_annotation_hub = importr("AnnotationHub")
double_brackets = ro.r("function(obj, idx){return(obj[[idx]])}")


class OrgDB:
    def __init__(self, species: str = "Homo sapiens"):
        self.species = species
        # self.db_lock = Manager().Lock()

    @property
    def db(self):
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
