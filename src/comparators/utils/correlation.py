from itertools import combinations
from typing import Tuple

import numpy as np
import pandas as pd
from numba import njit, prange
from rich.progress import track
from scipy import stats


@njit
def _delete_from_list(list, element):
    list2 = list.copy()
    list2.remove(element)
    return np.array(list2)


@njit(parallel=True)
def _calculate_correlations(
    activities: np.ndarray, class_combinations: np.ndarray, num_samples_from_second_class: int
) -> np.ndarray:
    n_seeds = activities.shape[0]
    n_layers = activities.shape[2]
    n_samples = activities.shape[3]

    n_correlation_samples = (
        n_seeds * len(class_combinations) * n_samples * num_samples_from_second_class
    )

    num_samples_from_second_class = (
        num_samples_from_second_class if num_samples_from_second_class < n_samples else n_samples
    )

    correlations = np.zeros((n_layers, n_correlation_samples))
    sample_indices = list(range(n_samples))

    for layer in prange(n_layers):
        sample_idx = 0
        for seed in range(n_seeds):
            for c1, c2 in class_combinations:
                for s1 in sample_indices:
                    for s2 in np.random.choice(
                        _delete_from_list(sample_indices, s1),
                        size=num_samples_from_second_class,
                        replace=False,
                    ):
                        correlations[layer, sample_idx] = np.corrcoef(
                            activities[seed, c1, layer, s1], activities[seed, c2, layer, s2]
                        )[0][1]
                        sample_idx += 1

    return correlations


def calculate_average_correlations(
    activities: np.ndarray,
    num_samples_from_second_class: int = 9,
    confidence_interval_percentile: float = 0.99,
) -> pd.DataFrame:

    n_classes = activities.shape[1]

    same_class_combinations = np.array([(c, c) for c in range(n_classes)])
    different_class_combinations = np.array(
        list(combinations(range(n_classes), 2)), dtype=np.int64
    )

    same_class_correlations = _calculate_correlations(
        activities, same_class_combinations, num_samples_from_second_class
    )
    different_class_correlations = _calculate_correlations(
        activities, different_class_combinations, num_samples_from_second_class
    )

    correlations = pd.DataFrame(
        index=["Same class", "Different class"],
        columns=pd.MultiIndex.from_product(
            [list(range(activities.shape[2])), ["Correlation", "Variance", "Error"]],
            names=["Layer", "Measurement"],
        ),
    )

    for layer in track(
        range(same_class_correlations.shape[0]),
        description="Calculating correlations for model...",
    ):

        correlations.loc["Same class", (layer, "Correlation")] = same_class_correlations[
            layer
        ].mean()

        if len(same_class_correlations[layer]) > 1:
            correlations.loc["Same class", (layer, "Variance")] = same_class_correlations[
                layer
            ].var(ddof=1)
            n_same_class_correlations = len(same_class_correlations[layer])
            confidence_interval = stats.t.interval(
                confidence_interval_percentile,
                n_same_class_correlations - 1,
                loc=correlations.loc["Same class", (layer, "Correlation")],
                scale=np.sqrt(
                    correlations.loc["Same class", (layer, "Variance")] / n_same_class_correlations
                ),
            )
            correlations.loc["Same class", (layer, "Error")] = (
                confidence_interval[1] - confidence_interval[0]
            ) / 2
        else:
            correlations.loc["Same class", (layer, "Variance")] = 0
            correlations.loc["Same class", (layer, "Error")] = 0

        correlations.loc["Different class", (layer, "Correlation")] = different_class_correlations[
            layer
        ].mean()

        if len(different_class_correlations[layer]) > 1:
            correlations.loc[
                "Different class", (layer, "Variance")
            ] = different_class_correlations[layer].var(ddof=1)
            n_different_class_correlations = len(different_class_correlations[layer])
            confidence_interval = stats.t.interval(
                confidence_interval_percentile,
                n_different_class_correlations - 1,
                loc=correlations.loc["Different class", (layer, "Correlation")],
                scale=np.sqrt(
                    correlations.loc["Different class", (layer, "Variance")]
                    / n_different_class_correlations
                ),
            )
            correlations.loc["Different class", (layer, "Error")] = (
                confidence_interval[1] - confidence_interval[0]
            ) / 2
        else:
            correlations.loc["Different class", (layer, "Variance")] = 0
            correlations.loc["Different class", (layer, "Error")] = 0

    return correlations
