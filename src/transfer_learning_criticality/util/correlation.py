from typing import Tuple
import pandas as pd
import numpy as np
from itertools import combinations
from scipy import stats
from numba import njit


@njit 
def _delete_from_list(list, element):
    list2 = list.copy()
    list2.remove(element)
    return np.array(list2)


@njit
def _calculate_average_correlations(activities:np.ndarray, class_combinations: np.ndarray, num_samples_from_second_class: int):

    n_samples = activities.shape[1]
    n_layers = activities.shape[2]
   
    average_correlations = np.zeros((n_layers, 3))
    sample_indices = list(range(n_samples))

    for l in range(n_layers):

        n_correlation_samples = n_samples * num_samples_from_second_class

        correlations = np.zeros((len(class_combinations), n_correlation_samples))

        for class_idx, (c1, c2) in enumerate(class_combinations):
            
            sample_idx = 0
            for s1 in sample_indices:
                for s2 in np.random.choice(_delete_from_list(sample_indices, s1), size=num_samples_from_second_class, replace=False):
                    correlations[class_idx, sample_idx] = np.corrcoef(activities[c1, s1, l], activities[c2, s2, l])[0][1]
                    sample_idx += 1

        average_correlations[l, 0] = correlations.mean()
        average_correlations[l, 1] = correlations.var()
        average_correlations[l, 2] = len(correlations)
        
    return average_correlations


def calculate_average_correlations(activities: pd.DataFrame, num_samples_from_second_class:int=1, confidence_interval_percentile:float=0.99) -> pd.DataFrame:

    class_indices = activities.index.unique(level="Class")
    layer_indices = activities.columns.unique(level="Layer")
    sample_indices = activities.index.unique(level="Sample")

    n_classes = len(class_indices)
    same_class_combinations = np.array([(c, c) for c in range(n_classes)])
    different_class_combinations = np.array(list(combinations(range(n_classes), 2)), dtype=np.int64)

    activities_array = activities.astype(np.float64).to_numpy().reshape((n_classes, len(sample_indices), len(layer_indices), -1))
    same_class_correlations = _calculate_average_correlations(activities_array, same_class_combinations, num_samples_from_second_class)
    different_class_correlations = _calculate_average_correlations(activities_array, different_class_combinations, num_samples_from_second_class)

    correlations = pd.DataFrame(index=["Same class", "Different class"], columns=pd.MultiIndex.from_product([layer_indices, ["Correlation", "Variance", "Error"]], names=["Layer", "Measurement"]))

    for l in range(same_class_correlations.shape[0]):

        correlations.loc["Same class", (l, "Correlation")] = same_class_correlations[l, 0]
        correlations.loc["Same class", (l, "Variance")] = same_class_correlations[l, 1]
        n_same_class_correlations = same_class_correlations[l, 2]
        confidence_interval = stats.t.interval(
            confidence_interval_percentile,
            n_same_class_correlations - 1,
            loc=correlations.loc["Same class", (l, "Correlation")],
            scale=np.sqrt(correlations.loc["Same class", (l, "Variance")] / n_same_class_correlations)
        )
        correlations.loc["Same class", (l, "Error")] = (confidence_interval[1] - confidence_interval[0]) / 2

        correlations.loc["Different class", (l, "Correlation")] = different_class_correlations[l, 0]
        correlations.loc["Different class", (l, "Variance")] = different_class_correlations[l, 1]
        n_different_class_correlations = different_class_correlations[l, 2]
        confidence_interval = stats.t.interval(
            confidence_interval_percentile,
            n_different_class_correlations - 1,
            loc=correlations.loc["Different class", (l, "Correlation")],
            scale=np.sqrt(correlations.loc["Different class", (l, "Variance")] / n_different_class_correlations)
        )
        correlations.loc["Different class", (l, "Error")] = (confidence_interval[1] - confidence_interval[0]) / 2

    return correlations