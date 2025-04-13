import json
from typing import Any

import numpy as np
import pandas as pd


class NpEncoder(json.JSONEncoder):
    """
    Custom JSON Encoder for handling NumPy and pandas objects.

    This encoder extends the standard JSON encoder to properly serialize
    NumPy and pandas data types, which are not natively serializable with
    the default JSON encoder.

    Examples:
        >>> import json
        >>> import numpy as np
        >>> from components.np_encoder import NpEncoder
        >>>
        >>> # Serialize NumPy arrays
        >>> data = {"array": np.array([1, 2, 3])}
        >>> json.dumps(data, cls=NpEncoder)
        '{"array": [1, 2, 3]}'
    """

    def default(self, obj: Any) -> Any:
        """
        Override the default method to handle NumPy and pandas types.

        Args:
            obj: The object to be serialized to JSON.

        Returns:
            A JSON-serializable representation of the object.

        Raises:
            TypeError: If the object type cannot be serialized to JSON.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_numpy()
        if isinstance(obj, pd.Series):
            return obj.to_numpy()
        return super(NpEncoder, self).default(obj)
