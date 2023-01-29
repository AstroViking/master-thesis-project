import pandas as pd
import numpy as np
from itertools import combinations_with_replacement
from scipy import stats


def calculate_average_correlations(activities: pd.DataFrame, num_samples_from_second_class:int=1, confidence_interval_percentile:float=0.99) -> pd.DataFrame:
    
    class_indices = activities.index.unique(level="Class")
    layer_indices = activities.columns.unique(level="Layer")
    sample_indices = activities.index.unique(level="Sample")

    class_combinations = list(combinations_with_replacement(class_indices, 2))

    correlations = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(class_combinations, names=["First class", "Second class"]), 
        columns=pd.MultiIndex.from_product([layer_indices, ["Correlation", "Variance", "Error"]], names=["Layer", "Measurement"])
    )

    n_samples = len(sample_indices) * num_samples_from_second_class

    for c1, c2 in class_combinations:
        for l in layer_indices:

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


    return correlations


def calculate_average_correlations_same_vs_different_class(correlations: pd.DataFrame, confidence_interval_percentile:float=0.99) -> pd.DataFrame:
    
    n_classes = len(correlations.index.unique(level="First class"))
    n_different_class_combinations = len(correlations.index) - n_classes

    combined_correlations = pd.DataFrame(index=["Same class", "Different class"], columns=correlations.columns.copy())
    same_class_mask = correlations.index.get_level_values("First class") == correlations.index.get_level_values("Second class")

    combined_correlations.loc["Same class", (slice(None), "Correlation")] = correlations.loc[same_class_mask].mean(axis=0)
    combined_correlations.loc["Same class", (slice(None), "Variance")] = correlations.loc[same_class_mask].var(axis=0, ddof=1)
    combined_correlations.loc["Different class", (slice(None), "Correlation")] = correlations.loc[~same_class_mask].mean(axis=0)
    combined_correlations.loc["Different class", (slice(None), "Variance")] = correlations.loc[~same_class_mask].var(axis=0, ddof=1)

    for l in combined_correlations.columns.unique(level="Layer"):

        confidence_interval = stats.t.interval(
            confidence_interval_percentile,
            n_classes - 1,
            loc=combined_correlations.loc["Same class", (l, "Correlation")],
            scale=np.sqrt(combined_correlations.loc["Same class", (l, "Variance")] / n_classes)
        )
        combined_correlations.loc["Same class", (slice(None), "Error")] = (confidence_interval[1] - confidence_interval[0]) / 2

        confidence_interval = stats.t.interval(
            confidence_interval_percentile,
            n_different_class_combinations - 1,
            loc=combined_correlations.loc["Different class", (l, "Correlation")],
            scale=np.sqrt(combined_correlations.loc["Different class", (l, "Variance")] / n_different_class_combinations)
        )
        combined_correlations.loc["Different class", (slice(None), "Error")] = (confidence_interval[1] - confidence_interval[0]) / 2

    return combined_correlations