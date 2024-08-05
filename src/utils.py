import contextvars
import logging
import traceback
from contextlib import contextmanager
from typing import Callable, Dict

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

logger = logging.getLogger(__name__)


@contextmanager
def context_wrapper():
    ctx = contextvars.copy_context()
    yield lambda func, *args, **kwargs: ctx.run(func, *args, **kwargs)


def run_func_dict(kwargs: Dict, func: Callable):
    logger.info(f"Starting execution of {func.__name__} with arguments: {kwargs}")
    try:
        with localconverter(ro.default_converter + pandas2ri.converter):
            with context_wrapper() as run_in_context:
                result = run_in_context(func, **kwargs)
                logger.info(
                    f"Successfully executed {func.__name__} with result: {result}"
                )
                return result
    except Exception as e:
        logger.error(f"Error occurred while executing {func.__name__}: {e}")
        logger.debug(traceback.print_exc())
