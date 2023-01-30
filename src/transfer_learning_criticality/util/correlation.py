from typing import Tuple
import pandas as pd
import numpy as np
from itertools import combinations_with_replacement
from scipy import stats


def calculate_average_correlations(activities: pd.DataFrame, num_samples_from_second_class:int=1, confidence_interval_percentile:float=0.99) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    class_indices = activities.index.unique(level="Class")
    layer_indices = activities.columns.unique(level="Layer")
    sample_indices = activities.index.unique(level="Sample")

    n_samples = len(sample_indices) * num_samples_from_second_class

    class_combinations = list(combinations_with_replacement(class_indices, 2))

    correlations = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(class_combinations, names=["First class", "Second class"]), 
        columns=pd.MultiIndex.from_product([layer_indices, ["Correlation", "Variance", "Error"]], names=["Layer", "Measurement"])
    )

    combined_correlations = pd.DataFrame(index=["Same class", "Different class"], columns=correlations.columns.copy())

    for l in layer_indices:

        same_class_correlations: np.ndarray = np.array([])
        different_class_correlations: np.ndarray = np.array([])

        for c1, c2 in class_combinations:
        
            sample_correlations: np.ndarray = np.zeros(shape=(len(sample_indices), num_samples_from_second_class))

            for s1 in sample_indices:
                for s2_idx, s2 in enumerate(np.random.choice(np.delete(sample_indices, s1), replace=False, size=num_samples_from_second_class)):
                    sample_correlations[s1, s2_idx] = np.corrcoef(activities.loc[(c1, s1), l].to_numpy().astype(np.float64), activities.loc[(c2, s2), l].to_numpy().astype(np.float64))[0][1]

            average_correlation = sample_correlations.mean()
            average_correlation_var = sample_correlations.var(ddof=1)

            confidence_interval = stats.t.interval(
                confidence_interval_percentile,
                n_samples - 1,
                loc=average_correlation,
                scale=np.sqrt(average_correlation_var / n_samples)
            )

            correlations.loc[(c1, c2), (l, "Correlation")] = average_correlation
            correlations.loc[(c1, c2), (l, "Variance")] = average_correlation_var
            correlations.loc[(c1, c2), (l, "Error")] = (confidence_interval[1] - confidence_interval[0]) / 2

            if c1 == c2:
                same_class_correlations = np.append(same_class_correlations, sample_correlations)
            
            else:
                different_class_correlations = np.append(different_class_correlations, sample_correlations)

        combined_correlations.loc["Same class", (l, "Correlation")] = same_class_correlations.mean()
        combined_correlations.loc["Same class", (l, "Variance")] = same_class_correlations.var(ddof=1)
        n_same_class_correlations = len(same_class_correlations)

        confidence_interval = stats.t.interval(
            confidence_interval_percentile,
            n_same_class_correlations - 1,
            loc=combined_correlations.loc["Same class", (l, "Correlation")],
            scale=np.sqrt(combined_correlations.loc["Same class", (l, "Variance")] / n_same_class_correlations)
        )
        combined_correlations.loc["Same class", (l, "Error")] = (confidence_interval[1] - confidence_interval[0]) / 2

        combined_correlations.loc["Different class", (l, "Correlation")] = different_class_correlations.mean()
        combined_correlations.loc["Different class", (l, "Variance")] = different_class_correlations.var(ddof=1)
        n_different_class_correlations = len(different_class_correlations)

        
        confidence_interval = stats.t.interval(
            confidence_interval_percentile,
            n_different_class_correlations - 1,
            loc=combined_correlations.loc["Different class", (l, "Correlation")],
            scale=np.sqrt(combined_correlations.loc["Different class", (l, "Variance")] / n_different_class_correlations)
        )
        combined_correlations.loc["Different class", (l, "Error")] = (confidence_interval[1] - confidence_interval[0]) / 2

    return correlations, combined_correlations


def calculate_combined_correlations_from_correlation_means(correlations: pd.DataFrame, confidence_interval_percentile:float=0.99) -> pd.DataFrame:
    
    n_classes = len(correlations.index.unique(level="First class"))
    n_different_class_combinations = len(correlations.index) - n_classes

    combined_correlations = pd.DataFrame(index=["Same class", "Different class"], columns=correlations.columns.copy())
    same_class_mask = correlations.index.get_level_values("First class") == correlations.index.get_level_values("Second class")

    combined_correlations.loc["Same class", (slice(None), "Correlation")] = correlations.loc[same_class_mask, (slice(None), "Correlation")].mean(axis=0).to_numpy()
    combined_correlations.loc["Same class", (slice(None), "Variance")] = correlations.loc[same_class_mask, (slice(None), "Correlation")].var(axis=0, ddof=1).to_numpy()
    combined_correlations.loc["Different class", (slice(None), "Correlation")] = correlations.loc[~same_class_mask, (slice(None), "Correlation")].mean(axis=0).to_numpy()
    combined_correlations.loc["Different class", (slice(None), "Variance")] = correlations.loc[~same_class_mask, (slice(None), "Correlation")].var(axis=0, ddof=1).to_numpy()

    for l in combined_correlations.columns.unique(level="Layer"):

        confidence_interval = stats.t.interval(
            confidence_interval_percentile,
            n_classes - 1,
            loc=combined_correlations.loc["Same class", (l, "Correlation")],
            scale=np.sqrt(combined_correlations.loc["Same class", (l, "Variance")] / n_classes)
        )
        combined_correlations.loc["Same class", (l, "Error")] = (confidence_interval[1] - confidence_interval[0]) / 2

        confidence_interval = stats.t.interval(
            confidence_interval_percentile,
            n_different_class_combinations - 1,
            loc=combined_correlations.loc["Different class", (l, "Correlation")],
            scale=np.sqrt(combined_correlations.loc["Different class", (l, "Variance")] / n_different_class_combinations)
        )

        combined_correlations.loc["Different class", (l, "Error")] = (confidence_interval[1] - confidence_interval[0]) / 2

    return combined_correlations