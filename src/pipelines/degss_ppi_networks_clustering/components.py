# source:
# https://github.com/jaumpedro214/posts/blob/main/ensamble_clustering/simlarity_matrix.py

from typing import Any, Iterable

import numpy as np
from scipy.spatial.distance import cdist


class ClusterSimilarityMatrix:
    def __init__(self) -> None:
        """
        Initialize a ClusterSimilarityMatrix object.

        This class is used to compute and store a similarity matrix for clustering
        algorithms. The similarity matrix is updated incrementally as new cluster
        assignments are provided.
        """
        self._is_fitted = False

    def fit(self, y_clusters: np.ndarray) -> "ClusterSimilarityMatrix":
        """
        Fit the similarity matrix with the given cluster assignments.

        Args:
            y_clusters (np.ndarray): Array of cluster assignments.

        Returns:
            ClusterSimilarityMatrix: The updated similarity matrix object.
        """
        if not self._is_fitted:
            self._is_fitted = True
            self.similarity = self.to_binary_matrix(y_clusters)
            return self

        self.similarity += self.to_binary_matrix(y_clusters)

    def to_binary_matrix(self, y_clusters: np.ndarray) -> np.ndarray:
        """
        Convert cluster assignments to a binary similarity matrix.

        Args:
            y_clusters (np.ndarray): Array of cluster assignments.

        Returns:
            np.ndarray: Binary similarity matrix.
        """
        y_reshaped = np.expand_dims(y_clusters, axis=-1)
        return (cdist(y_reshaped, y_reshaped, "cityblock") == 0).astype(int)


class EnsembleCustering:
    def __init__(
        self,
        base_estimators: Iterable[Any],
        aggregator: Any,
        distances: bool = False,
    ) -> None:
        """
        Initialize an EnsembleCustering object.

        Args:
            base_estimators (Iterable[Any]): List of base clustering estimators.
            aggregator (Any): Aggregator clustering method.
            distances (bool): Whether to compute distances instead of similarities.
        """
        self.base_estimators = base_estimators
        self.aggregator = aggregator
        self.distances = distances

    def fit(self, X: np.ndarray) -> None:
        """
        Fit the ensemble clustering model to the data.

        Args:
            X (np.ndarray): Input data for clustering.
        """
        X_ = X.copy()

        clt_sim_matrix = ClusterSimilarityMatrix()
        for model in self.base_estimators:
            clt_sim_matrix.fit(model.fit_predict(X=X_))

        sim_matrix = clt_sim_matrix.similarity
        self.cluster_matrix = sim_matrix / sim_matrix.diagonal()

        if self.distances:
            self.cluster_matrix = np.abs(
                np.log(self.cluster_matrix + 1e-8)
            )  # Avoid log(0)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the model and predict cluster assignments.

        Args:
            X (np.ndarray): Input data for clustering.

        Returns:
            np.ndarray: Predicted cluster assignments.
        """
        self.fit(X)
        y = self.aggregator.fit_predict(self.cluster_matrix)
        return y
