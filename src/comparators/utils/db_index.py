from itertools import combinations
from typing import Tuple

import numpy as np
import pandas as pd
from numba import njit
from scipy import stats
from sklearn.metrics import davies_bouldin_score


def calculate_davies_bouldin_index(
    activities: np.ndarray,
    confidence_interval_percentile: float = 0.99,
) -> pd.DataFrame:

    n_seeds = activities.shape[0]
    n_classes = activities.shape[1]
    n_layers = activities.shape[2]
    n_samples = activities.shape[3]
    n_neurons = activities.shape[4]

    db_indexes = pd.DataFrame(
        columns=pd.MultiIndex.from_product(
            [list(range(n_layers)), ["DB Index", "Variance", "Error"]],
            names=["Layer", "Measurement"],
        ),
    ).astype(np.float64)

    for layer in range(n_layers):

        db_index_samples = np.zeros(n_seeds)
        for seed in range(n_seeds):
            db_index_samples[seed] = davies_bouldin_score(
                activities[seed, :, layer].reshape(-1, n_neurons),
                np.repeat([*range(n_classes)], n_samples),
            )

        db_indexes.loc[(layer, "DB Index")] = db_index_samples.mean()

        if len(db_index_samples) > 1:
            db_indexes.loc[(layer, "Variance")] = db_index_samples.var(ddof=1)
            confidence_interval = stats.t.interval(
                confidence_interval_percentile,
                n_seeds - 1,
                loc=db_indexes.loc[(layer, "DB Index")],
                scale=np.sqrt(db_indexes.loc[(layer, "Variance")] / n_seeds),
            )
            db_indexes.loc[(layer, "Error")] = (
                confidence_interval[1] - confidence_interval[0]
            ) / 2
        else:
            db_indexes.loc[(layer, "Variance")] = 0
            db_indexes.loc[(layer, "Error")] = 0

    return db_indexes
