import copy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import clustbench
import genieclust
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_mutual_info_score,
    fowlkes_mallows_score,
)
from typing import Union, Dict, List, Optional, Tuple, Literal
from io import StringIO


class ClusteringEvaluator:

    # Stores scores from evaluation in dataframe with new metric columns
    results_df: pd.DataFrame = pd.DataFrame(
        {
            "battery": pd.Series(dtype="str"),
            "dataset": pd.Series(dtype="str"),
            "method": pd.Series(dtype="str"),
            "labels": pd.Series(dtype="str"),
            "n_clusters": pd.Series(dtype="int"),
            "rand_score": pd.Series(dtype="float"),
            "silhouette_score": pd.Series(dtype="float"),
            "calinski_harabasz_score": pd.Series(dtype="float"),
            "davies_bouldin_score": pd.Series(dtype="float"),
            "adjusted_mutual_info_score": pd.Series(dtype="float"),
            "fowlkes_mallows_score": pd.Series(dtype="float"),
            "NCA": pd.Series(dtype="float"),
        }
    )

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the ClusteringEvaluator with the path to the clustering data.

        Args:
            data_path: Path to the clustering data directory
        """
        if data_path is not None:
            self.data_path = os.path.abspath(data_path)
        else:
            self.data_path = os.path.abspath("clustering-data-v1")
        self.methods = ["kmeans", "dbscan", "agglomerative", "genie"]
        self.g = [0.1, 0.3, 0.5, 0.7, 0.9]
        self.linkage = ["ward", "single", "complete", "average"]
        self.eps = [0.2]
        self.min_samples = [5]

    def load_data(self, battery: str, dataset: str) -> None:
        """
        Load a specific dataset from the given battery.
        """
        self.b = clustbench.load_dataset(battery, dataset, path=self.data_path)
        self.data = self.b.data
        self.labels = self.b.labels
        self.n_clusters = self.b.n_clusters

    def get_clusterer(
        self, method: str, **kwargs
    ) -> Union[KMeans, DBSCAN, AgglomerativeClustering, genieclust.Genie]:
        """
        Get the clustering algorithm based on the method name.

        Args:
            method: Name of the clustering method

        Returns:
            An instance of the clustering algorithm
        """

        if method == "kmeans":
            return KMeans()
        elif method == "dbscan":
            # DBSCAN requires eps and min_samples parameters
            eps = kwargs.get("eps", 0.2)
            min_samples = kwargs.get("min_samples", 5)
            return DBSCAN(eps=eps, min_samples=min_samples)
        elif method == "agglomerative":
            linkage = kwargs.get("linkage", "ward")
            return AgglomerativeClustering(linkage=linkage)
        elif method == "genie":
            gini_threshold = kwargs.get("gini_threshold", 0.3)
            return genieclust.Genie(gini_threshold=gini_threshold)
        else:
            raise ValueError(f"Unknown method: {method}")

    def custom_fit_predict_many(self, clusterer, data, n_clusters):
        """
        Custom implementation of fit_predict_many that handles duplicate cluster numbers
        by returning a dictionary with unique keys for each partition.

        Args:
            clusterer: Clustering algorithm instance
            data: Dataset to cluster
            n_clusters: List of cluster numbers

        Returns:
            Dictionary mapping unique keys to cluster assignments
        """
        results = {}

        # Create a unique key for each n_cluster value by appending an index for duplicates
        cluster_counts = {}

        for i, k in enumerate(n_clusters):
            k = int(k)  # Ensure k is an integer

            # Use original number as key for first occurrence
            if k not in cluster_counts:
                key = str(k)
                cluster_counts[k] = 1
            else:
                # For duplicates, append a suffix to the key: k_1, k_2, etc.
                key = f"{k}_{cluster_counts[k]}"
                cluster_counts[k] += 1

            # Clone the clusterer and set n_clusters if it has this parameter
            clone_clusterer = copy.deepcopy(clusterer)
            if hasattr(clusterer, "n_clusters"):
                clone_clusterer.n_clusters = k
            pred_labels = clone_clusterer.fit_predict(data) + 1

            results[key] = pred_labels

        return results

    def do_all(self, battery: str, dataset: str, plot: bool) -> None:
        """
        Run all defined clustering methods on a dataset and store the evaluation results.

        Args:
            battery: Name of the battery (e.g., 'wut')
            dataset: Name of the dataset (e.g., 'x2')
            plot: Whether to plot the clustering results
        """
        self.load_data(battery, dataset)
        print(f"Loaded data for {battery} - {dataset}")

        method_params = []
        # Create a list of (method, params_dict) tuples to iterate over
        for method in self.methods:
            if method == "genie":
                # For genie, use different gini_threshold values
                for g in self.g:
                    method_params.append((method, {"gini_threshold": g}))
            elif method == "agglomerative":
                # For agglomerative, use different linkage values
                for linkage in self.linkage:
                    method_params.append((method, {"linkage": linkage}))
            elif method == "dbscan":
                # For dbscan, use different eps and min_samples values
                for eps in self.eps:
                    for min_samples in self.min_samples:
                        method_params.append(
                            (method, {"eps": eps, "min_samples": min_samples})
                        )
            else:
                # For other methods, use default params
                method_params.append((method, {}))

        # Plot true labels
        if plot:
            # Dynamically adjust figure width based on number of clusters
            n_cols = len(self.n_clusters)
            num_rows = len(method_params) + 1  # +1 for the true labels row
            fig_width = max(10, 5 * n_cols)  # Base width of 5 per column, minimum 10
            fig_height = max(10, 4 * num_rows)  # Dynamic height based on rows
            plt.figure(figsize=(fig_width, fig_height))

            for i in range(len(self.labels)):
                plt.subplot(num_rows, n_cols, i + 1)
                genieclust.plots.plot_scatter(
                    self.data,
                    labels=self.labels[i] - 1,
                    axis="equal",
                    title=f"True Labels (k = {self.n_clusters[i]}) ",
                )

        for iter, (method, params) in enumerate(method_params):
            # Results should be stored in a dictionary with keys as the number of clusters and list of labels as values
            clusterer = self.get_clusterer(method, **params)
            results = self.custom_fit_predict_many(
                clusterer, self.data, self.n_clusters
            )

            for i, k_key in enumerate(results):
                k = (
                    int(k_key.split("_")[0])
                    if isinstance(k_key, str) and "_" in k_key
                    else int(k_key)
                )

                if plot:
                    ax = plt.subplot(num_rows, n_cols, (iter + 1) * n_cols + i + 1)

                    if method == "genie" and "gini_threshold" in params:
                        title = f"{method.capitalize()} (g={params['gini_threshold']}) k={k}"
                    elif method == "agglomerative" and "linkage" in params:
                        title = f"{method.capitalize()} ({params['linkage']}) k={k}"
                    elif method == "dbscan":
                        title = f"{method.capitalize()} (eps={params['eps']}, min_samples={params['min_samples']})"
                    else:
                        title = f"{method.capitalize()} Labels (k = {k})"

                    genieclust.plots.plot_scatter(
                        self.data, labels=results[k_key] - 1, axis="equal", title=title
                    )
                    if int(k) < 6:
                        confusion_matrix = (
                            genieclust.compare_partitions.confusion_matrix(
                                self.labels[i], results[k_key]
                            )
                        )
                        cm_str = StringIO()
                        np.savetxt(
                            cm_str,
                            confusion_matrix,
                            fmt="%d",
                            delimiter=" | ",
                            footer="\nTrue \\\\ Pred",
                            comments="",
                        )
                        cm_text = cm_str.getvalue()
                        ax.text(
                            0.95,
                            0.05,
                            cm_text,
                            transform=ax.transAxes,
                            fontsize=8,
                            ha="right",
                            va="bottom",
                            bbox=dict(facecolor="white", alpha=0.5, edgecolor="black"),
                        )

                if not self.check_if_exists(
                    battery, dataset, method, int(k), params, i
                ):
                    # Check if internal scores can be calculated (requires > 1 cluster)
                    can_calc_internal_scores = len(np.unique(results[k_key])) > 1

                    df = pd.DataFrame(
                        {
                            "battery": battery,
                            "dataset": dataset,
                            "method": method,
                            "labels": f"labels{i}",
                            "n_clusters": int(k),
                            "rand_score": adjusted_rand_score(
                                self.labels[i], results[k_key]
                            ),
                            "silhouette_score": (
                                silhouette_score(self.data, results[k_key])
                                if can_calc_internal_scores
                                else float("nan")
                            ),
                            "calinski_harabasz_score": (
                                calinski_harabasz_score(self.data, results[k_key])
                                if can_calc_internal_scores
                                else float("nan")
                            ),
                            "davies_bouldin_score": (
                                davies_bouldin_score(self.data, results[k_key])
                                if can_calc_internal_scores
                                else float("nan")
                            ),
                            "adjusted_mutual_info_score": adjusted_mutual_info_score(
                                self.labels[i], results[k_key]
                            ),
                            "fowlkes_mallows_score": fowlkes_mallows_score(
                                self.labels[i], results[k_key]
                            ),
                            "NCA": genieclust.compare_partitions.normalized_clustering_accuracy(
                                self.labels[i], results[k_key]
                            ),
                            "params": "default",
                        },
                        index=[0],
                    )

                    if params is not None:
                        for key, value in params.items():
                            df[key] = value

                    self.results_df = pd.concat(
                        [self.results_df, df], ignore_index=True
                    )
                else:
                    print(
                        f"Skipping existing result: {battery}-{dataset}, {method}, k={k}, params={params}"
                    )

        if plot:
            plt.tight_layout()
            plt.show()

    def evaluate_single_method(
        self, battery: str, dataset: str, method: str, plot: bool = False, **kwargs
    ) -> None:
        """
        Evaluate a single clustering method with custom parameters on a specific dataset.

        Args:
            battery: Name of the battery (e.g., 'wut')
            dataset: Name of the dataset (e.g., 'x2')
            method: Clustering method ('kmeans', 'dbscan', 'agglomerative', 'genie')
            plot: Whether to plot the clustering results
            **kwargs: Custom parameters for the clustering method
        """
        if not hasattr(self, "data") or (
            hasattr(self, "b")
            and (self.b.battery != battery or self.b.dataset != dataset)
        ):
            self.load_data(battery, dataset)
            print(f"Loaded data for {battery} - {dataset}")

        clusterer = self.get_clusterer(method, **kwargs)
        print(f"clusterer: {clusterer}")
        results = self.custom_fit_predict_many(clusterer, self.data, self.n_clusters)

        if plot:
            n_cols = len(self.n_clusters)
            fig_width = max(10, 5 * n_cols)
            plt.figure(figsize=(fig_width, 4 * 2))

            for i, k in enumerate(self.n_clusters):
                plt.subplot(2, n_cols, i + 1)
                genieclust.plots.plot_scatter(
                    self.data,
                    labels=self.labels[i] - 1,
                    axis="equal",
                    title=f"True Labels (k = {k})",
                )

            for i, k_key in enumerate(results):
                k = (
                    int(k_key.split("_")[0])
                    if isinstance(k_key, str) and "_" in k_key
                    else int(k_key)
                )
                ax = plt.subplot(2, len(results), len(results) + i + 1)

                if method == "genie" and "gini_threshold" in kwargs:
                    title = (
                        f"{method.capitalize()} (g={kwargs['gini_threshold']}) k={k}"
                    )
                elif method == "agglomerative" and "linkage" in kwargs:
                    title = f"{method.capitalize()} ({kwargs['linkage']}) k={k}"
                elif method == "dbscan":
                    title = f"{method.capitalize()} (eps={kwargs.get('eps', 0.5)}, min_samples={kwargs.get('min_samples', 5)})"
                else:
                    title = f"{method.capitalize()} Labels (k = {k})"

                genieclust.plots.plot_scatter(
                    self.data, labels=results[k_key] - 1, axis="equal", title=title
                )

                if int(k) < 6:
                    confusion_matrix = genieclust.compare_partitions.confusion_matrix(
                        self.labels[i], results[k_key]
                    )
                    cm_str = StringIO()
                    np.savetxt(
                        cm_str,
                        confusion_matrix,
                        fmt="%d",
                        delimiter=" | ",
                        footer="\nTrue \\\\ Pred",
                        comments="",
                    )
                    cm_text = cm_str.getvalue()
                    ax.text(
                        0.95,
                        0.05,
                        cm_text,
                        transform=ax.transAxes,
                        fontsize=8,
                        ha="right",
                        va="bottom",
                        bbox=dict(facecolor="white", alpha=0.5, edgecolor="black"),
                    )

            plt.tight_layout()
            plt.show()

        for i, k_key in enumerate(results):
            k = (
                int(k_key.split("_")[0])
                if isinstance(k_key, str) and "_" in k_key
                else int(k_key)
            )

            if not self.check_if_exists(battery, dataset, method, int(k), kwargs, i):
                # Check if internal scores can be calculated (requires > 1 cluster)
                can_calc_internal_scores = len(np.unique(results[k_key])) > 1

                df = pd.DataFrame(
                    {
                        "battery": battery,
                        "dataset": dataset,
                        "method": method,
                        "labels": f"labels{i}",
                        "n_clusters": int(k),
                        "rand_score": adjusted_rand_score(
                            self.labels[i], results[k_key]
                        ),
                        "silhouette_score": (
                            silhouette_score(self.data, results[k_key])
                            if can_calc_internal_scores
                            else float("nan")
                        ),
                        "calinski_harabasz_score": (
                            calinski_harabasz_score(self.data, results[k_key])
                            if can_calc_internal_scores
                            else float("nan")
                        ),
                        "davies_bouldin_score": (
                            davies_bouldin_score(self.data, results[k_key])
                            if can_calc_internal_scores
                            else float("nan")
                        ),
                        "adjusted_mutual_info_score": adjusted_mutual_info_score(
                            self.labels[i], results[k_key]
                        ),
                        "fowlkes_mallows_score": fowlkes_mallows_score(
                            self.labels[i], results[k_key]
                        ),
                        "NCA": genieclust.compare_partitions.normalized_clustering_accuracy(
                            self.labels[i], results[k_key]
                        ),
                        "params": "custom",
                    },
                    index=[0],
                )

                for key, value in kwargs.items():
                    df[key] = value

                self.results_df = pd.concat([self.results_df, df], ignore_index=True)
            else:
                print(
                    f"Skipping existing result: {battery}-{dataset}, {method}, k={k}, params={kwargs}"
                )

    def check_if_exists(
        self,
        battery,
        dataset,
        method,
        n_clusters=None,
        params_dict=None,
        labels_index=None,
    ):
        """
        Check if a result with the given parameters already exists in the results dataframe.
        """
        if len(self.results_df) == 0:
            return False

        mask = (
            (self.results_df["battery"] == battery)
            & (self.results_df["dataset"] == dataset)
            & (self.results_df["method"] == method)
        )

        if n_clusters is not None:
            mask &= self.results_df["n_clusters"] == n_clusters

        if labels_index is not None:
            mask &= self.results_df["labels"] == f"labels{labels_index}"

        # Distinguish between default runs (do_all) and custom runs (evaluate_single_method)
        params_type = "custom" if params_dict and any(params_dict) else "default"
        mask &= self.results_df["params"] == params_type

        if params_dict:
            for param_name, param_value in params_dict.items():
                if param_name in self.results_df.columns:
                    # Handle potential NaN values in the DataFrame column
                    mask &= (
                        self.results_df[param_name].fillna("__NONE__") == param_value
                    )
                else:
                    # If the parameter column doesn't exist, this entry can't be a match
                    return False

        return mask.any()

    def save_results(self, filename: str = "results.csv") -> None:
        """
        Save the results dataframe to a CSV file.
        """
        if len(self.results_df) == 0:
            print("No results to save.")
            return

        self.results_df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
