import logging
from typing import Callable, Dict

from rich.console import Console

console = Console()


def run_func_dict(kwargs: Dict, func: Callable):
    try:
        return func(**kwargs)
    except Exception as e:
        logging.error(e)
        console.print_exception(max_frames=20)
        return None
