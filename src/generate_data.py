import random
from sklearn.datasets import make_classification
import numpy as np


def generate_column_partition(n, k, random_state, min_first=2):
    if k < 1:
        raise ValueError("Number of categories must be at least 1.")
    if n < min_first:
        raise ValueError("Total must be at least as large as the minimum first value.")

    rng = random.Random(random_state)

    min_first += 1
    remaining = n - min_first + 1

    # Generate k-1 non-negative integers that sum to `remaining`
    # This is done by generating (k-2) cut points between 0 and remaining,
    # and sorting them to form the segments
    cuts = sorted(rng.sample(range(remaining + k - 1), k - 1))
    partition = [cuts[0]]
    for i in range(1, len(cuts)):
        partition.append(cuts[i] - cuts[i - 1])
    partition.append(remaining + k - 1 - cuts[-1])

    # Subtract 1 from each (stars and bars method)
    partition = [x - 1 for x in partition]

    # Prepend the fixed minimum for the first category
    partition[0] += min_first

    return partition


def generate_dataset(n_samples=1000, n_features=20, random_state=213):

    partition = generate_column_partition(n_features, 5, random_state=random_state)

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=partition[0],
        n_redundant=partition[1],
        n_repeated=partition[2] + partition[3],
        n_clusters_per_class=2,
        class_sep=1.0,
        random_state=random_state,
        shuffle=False,
    )

    correlated_index_start = partition[0] + partition[1]
    noise_index_start = correlated_index_start + partition[2]

    for i in range(partition[2]):
        index = correlated_index_start + i
        X[:, index] += np.random.normal(loc=0.0, scale=0.2, size=n_samples)

    for i in range(partition[3]):
        index = noise_index_start + i
        X[:, index] += np.random.uniform(low=2.0, high=2.0, size=n_samples)

    labels = ["informative", "redundant", "correlated", "noise", "irrelevant"]
    idx = np.cumsum([0] + partition)
    feature_types = {
        "informative": list(range(idx[0], idx[1])),
        "redundant": list(range(idx[1], idx[2])),
        "correlated": list(range(idx[2], idx[3])),
        "noise": list(range(idx[3], idx[4])),
        "pure_noise": list(range(idx[4], idx[5])),
    }

    print(partition)
    print()

    for key, val in feature_types.items():
        print(key, val)
    return X, y, feature_types
