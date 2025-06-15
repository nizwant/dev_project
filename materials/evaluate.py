import copy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import clustbench
import genieclust
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from typing import Union, Dict, List, Optional, Tuple, Literal
from io import StringIO

class ClusteringEvaluator:
    
    # stores scores from evaluation in dataframe with columns: battery, dataset, method, rand_score, silhoutte score, accuracy score
    results_df:pd.DataFrame = pd.DataFrame({
        'battery': pd.Series(dtype='str'),
        'dataset': pd.Series(dtype='str'),
        'method': pd.Series(dtype='str'),
        'labels': pd.Series(dtype='str'),
        'n_clusters': pd.Series(dtype='int'),
        'rand_score': pd.Series(dtype='float'),
        'silhouette_score': pd.Series(dtype='float'),
        'NCA': pd.Series(dtype='float'),
    })


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
        self.methods = ['kmeans', 'dbscan', 'agglomerative', 'genie']
        self.g = [0.1, 0.3, 0.5, 0.7, 0.9]
        self.linkage = ['ward', 'single', 'complete', 'average']
        self.eps = [0.2]
        self.min_samples = [5]
    
    def load_data(self, battery: str, dataset: str) -> Tuple[np.ndarray, np.ndarray]:
        self.b = clustbench.load_dataset(battery, dataset, path=self.data_path)
        self.data = self.b.data
        self.labels = self.b.labels
        self.n_clusters = self.b.n_clusters

    def get_clusterer(self, method: str, **kwargs) -> Union[KMeans, DBSCAN, AgglomerativeClustering]:
        """
        Get the clustering algorithm based on the method name.
        
        Args:
            method: Name of the clustering method
        
        Returns:
            An instance of the clustering algorithm
        """

        if method == 'kmeans':
            return KMeans()
        elif method == 'dbscan':
            # DBSCAN requires eps and min_samples parameters
            eps = kwargs.get('eps', 0.2)
            min_samples = kwargs.get('min_samples', 5)
            return DBSCAN(eps=eps, min_samples=min_samples)
        elif method == 'agglomerative':
            linkage = kwargs.get('linkage', 'ward')
            return AgglomerativeClustering(linkage=linkage)
        elif method == 'genie':
            gini_threshold = kwargs.get('gini_threshold', 0.3)
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
            if hasattr(clusterer, 'n_clusters'):
                clone_clusterer.n_clusters = k
            pred_labels = clone_clusterer.fit_predict(data) + 1
    
            results[key] = pred_labels
        
        return results

    def do_all(self, battery: str, dataset: str, plot:bool) -> None:

        self.load_data(battery, dataset)
        print(f"Loaded data for {battery} - {dataset}")

        method_params = []
        # Create a list of (method, params_dict) tuples to iterate over
        for method in self.methods:
            if method == 'genie':
                # For genie, use different gini_threshold values
                for g in self.g:
                    method_params.append((method, {'gini_threshold': g}))
            elif method == 'agglomerative':
                # For agglomerative, use different linkage values
                for linkage in self.linkage:
                    method_params.append((method, {'linkage': linkage}))
            elif method == 'dbscan':
                # For dbscan, use different eps and min_samples values
                for eps in self.eps:
                    for min_samples in self.min_samples:
                        method_params.append((method, {'eps': eps, 'min_samples': min_samples}))
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
                    labels=self.labels[i]-1,
                    axis='equal',
                    title=f"True Labels (k = {self.n_clusters[i]}) ",
                )

        for iter, (method, params) in enumerate(method_params):
            # Results should be stored in a dictionary with keys as the number of clusters and list of labels as values

            # create a clusterer
            clusterer = self.get_clusterer(method, **params)
            results = self.custom_fit_predict_many(clusterer, self.data, self.n_clusters)
            
            for i, k_key in enumerate(results):

                if isinstance(k_key, str) and '_' in k_key:
                    k = int(k_key.split('_')[0])
                else:
                    k = k_key
                                
                # plot results vs true labels
                if plot:
                    # Update subplot position using num_rows instead of len(self.methods)+1
                    ax = plt.subplot(num_rows, n_cols, (iter+1) * len(results) + i + 1)
                    
                    # Create method title with parameters
                    if method == 'genie' and 'gini_threshold' in params:
                        title = f"{method.capitalize()} (g={params['gini_threshold']}) k={k}"
                    elif method == 'agglomerative' and 'linkage' in params:
                        title = f"{method.capitalize()} ({params['linkage']}) k={k}"
                    elif method == 'dbscan':
                        title = f"{method.capitalize()} (eps={params['eps']}, min_samples={params['min_samples']})"
                    else:
                        title = f"{method.capitalize()} Labels (k = {k})"
                    
                    genieclust.plots.plot_scatter(
                        self.data,
                        labels=results[k_key]-1,
                        axis='equal',
                        title=title,
                    )
                    # If number of clusters is greather than 5 then do not plot confusion matrix
                    if int(k) < 6:
                        # The rest of your confusion matrix code remains the same
                        confusion_matrix = genieclust.compare_partitions.confusion_matrix(
                            self.labels[i], results[k_key]
                        )
                        cm_str = StringIO()
                        np.savetxt(cm_str, confusion_matrix, fmt='%d', delimiter=' | ', footer="\nTrue \\\\ Pred", comments = '')
                        cm_text = cm_str.getvalue()
                        ax.text(
                            0.95, 0.05, cm_text,
                            transform=ax.transAxes,
                            fontsize=8,
                            ha='right', va='bottom',
                            bbox=dict(facecolor='white', alpha=0.5, edgecolor='black')
                        )
                # Check if this configuration already exists
                if not self.check_if_exists(battery, dataset, method, int(k), params, i):
                    df = pd.DataFrame({
                    "battery": battery,
                    "dataset": dataset,
                    "method": method,
                    "labels": f'labels{i}',
                    "n_clusters": int(k),  # Store the original cluster number
                    "rand_score": adjusted_rand_score(self.labels[i], results[k_key]),
                    "silhouette_score": silhouette_score(self.data, results[k_key]) if len(np.unique(results[k_key])) > 1 else float('nan'),
                    # "NCA": clustbench.get_score(self.labels[i], results[k])
                    "NCA": genieclust.compare_partitions.normalized_clustering_accuracy(
                        self.labels[i], results[k_key]),
                    'params': 'default',
                }, index=[0])
                    if params is not None:
                        for key, value in params.items():
                            df[key] = value

                    self.results_df = pd.concat([self.results_df, df], ignore_index=True)
                else:
                    print(f"Skipping existing result: {battery}-{dataset}, {method}, k={k}, params={params}")

        if plot:
            plt.tight_layout()
            plt.show()

    def evaluate_single_method(self, battery: str, dataset: str, method: str, plot: bool = False, **kwargs) -> None:
        """
        Evaluate a single clustering method with custom parameters on a specific dataset.
        
        Args:
            battery: Name of the battery (e.g., 'wut')
            dataset: Name of the dataset (e.g., 'x2')
            method: Clustering method ('kmeans', 'dbscan', 'agglomerative', 'genie')
            plot: Whether to plot the clustering results
            **kwargs: Custom parameters for the clustering method
        """
        # Load data if not already loaded or if different dataset is requested
        if not hasattr(self, 'data') or (hasattr(self, 'b') and 
                                        (self.b.battery != battery or self.b.dataset != dataset)):
            self.load_data(battery, dataset)
            print(f"Loaded data for {battery} - {dataset}")
        
        # Get the clusterer with custom parameters
        clusterer = self.get_clusterer(method, **kwargs)

        print(f"clusterer: {clusterer}")
        
        # Get clustering results using the same method as do_all
        results = self.custom_fit_predict_many(clusterer, self.data, self.n_clusters)
        
        # Plot results if requested
        if plot:
            n_cols = len(self.n_clusters)
            fig_width = max(10, 5 * n_cols)  # Base width of 5 per column, minimum 5
            plt.figure(figsize=(fig_width, 4*2))
            
            # Plot true labels first
            for i, k in enumerate(self.n_clusters):
                plt.subplot(2, n_cols, i + 1)
                genieclust.plots.plot_scatter(
                    self.data,
                    labels=self.labels[i] - 1,
                    axis='equal',
                    title=f"True Labels (k = {k})",
                )
            
            # Plot predicted labels - make consistent with do_all method
            for i, k_key in enumerate(results):
                # Extract original k value if it's a string key with format "k_n"
                if isinstance(k_key, str) and '_' in k_key:
                    k = int(k_key.split('_')[0])
                else:
                    k = k_key
                    
                ax = plt.subplot(2, len(results), len(results) + i + 1)
                
                # Create method title with parameters
                if method == 'genie' and 'gini_threshold' in kwargs:
                    title = f"{method.capitalize()} (g={kwargs['gini_threshold']}) k={k}"
                elif method == 'agglomerative' and 'linkage' in kwargs:
                    title = f"{method.capitalize()} ({kwargs['linkage']}) k={k}"
                elif method == 'dbscan':
                    title = f"{method.capitalize()} (eps={kwargs.get('eps', 0.5)}, min_samples={kwargs.get('min_samples', 5)})"
                else:
                    title = f"{method.capitalize()} Labels (k = {k})"
                
                genieclust.plots.plot_scatter(
                    self.data,
                    labels=results[k_key] - 1,
                    axis='equal',
                    title=title,
                )
                
                if int(k) < 6:
                    # The rest of your confusion matrix code remains the same
                    confusion_matrix = genieclust.compare_partitions.confusion_matrix(
                        self.labels[i], results[k_key]
                    )
                    cm_str = StringIO()
                    np.savetxt(cm_str, confusion_matrix, fmt='%d', delimiter=' | ', footer="\nTrue \\\\ Pred", comments = '')
                    cm_text = cm_str.getvalue()
                    ax.text(
                        0.95, 0.05, cm_text,
                        transform=ax.transAxes,
                        fontsize=8,
                        ha='right', va='bottom',
                        bbox=dict(facecolor='white', alpha=0.5, edgecolor='black')
                    )
            
            plt.tight_layout()
            plt.show()
        
        # Store results in dataframe
        for i, k_key in enumerate(results):
            # Extract original k value if it's a string key
            if isinstance(k_key, str) and '_' in k_key:
                k = int(k_key.split('_')[0])
            else:
                k = k_key
                
            # Check if this configuration already exists
            if not self.check_if_exists(battery, dataset, method, int(k), kwargs, i):
                df = pd.DataFrame({
                    "battery": battery,
                    "dataset": dataset,
                    "method": method,
                    "labels": f'labels{i}',
                    "n_clusters": int(k),  # Store the original cluster number
                    "rand_score": adjusted_rand_score(self.labels[i], results[k_key]),
                    "silhouette_score": silhouette_score(self.data, results[k_key]) if len(np.unique(results[k_key])) > 1 else float('nan'),
                    "NCA": genieclust.compare_partitions.normalized_clustering_accuracy(
                        self.labels[i], results[k_key]),
                    'params': 'custom',
                }, index=[0])
                
                # Add all custom parameters to the dataframe
                for key, value in kwargs.items():
                    df[key] = value
                
                self.results_df = pd.concat([self.results_df, df], ignore_index=True)
            else:
                print(f"Skipping existing result: {battery}-{dataset}, {method}, k={k}, params={kwargs}")

    def check_if_exists(self, battery, dataset, method, n_clusters=None, params_dict=None, labels_index=None):
        """
        Check if a result with the given parameters already exists in the results dataframe.
        
        Args:
            battery: Battery name
            dataset: Dataset name
            method: Clustering method
            n_clusters: Number of clusters (optional)
            params_dict: Dictionary of additional parameters
            labels_index: Specific labels index (optional)
            
        Returns:
            Boolean indicating whether the combination exists
        """
        # Check if the dataframe is empty
        if len(self.results_df) == 0:
            return False
            
        # Start with basic filters
        mask = (
            (self.results_df['battery'] == battery) &
            (self.results_df['dataset'] == dataset) &
            (self.results_df['method'] == method)
        )
        
        # Add n_clusters filter if provided
        if n_clusters is not None:
            mask &= (self.results_df['n_clusters'] == n_clusters)

        # Check the specific labels index if provided
        if labels_index is not None:
            mask &= (self.results_df['labels'] == f'labels{labels_index}')
        # Add params type filter
        if params_dict is None:
            mask &= (self.results_df['params'] == 'default')
        else:
            mask &= (self.results_df['params'] == 'custom')
            
            # If we have params, filter by each parameter
            for param_name, param_value in params_dict.items():
                if param_name in self.results_df.columns:
                    mask &= (self.results_df[param_name] == param_value)
        
        # Return True if ANY row matches all these conditions
        return mask.any()

    def save_results(self, filename: str = "results.csv") -> None:
        """
        Save the results dataframe to a CSV file.
        
        Args:
            filename: Name of the output CSV file
        """
        # Ensure the directory exists
        if len(self.results_df) == 0:
            print("No results to save.")
            return

        self.results_df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")