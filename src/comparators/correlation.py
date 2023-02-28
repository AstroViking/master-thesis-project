from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from pytorch_lightning import LightningDataModule, LightningModule
from rich.progress import track

import src.figures as fig

from .. import utils
from ..models.inspectable import InspectableModule
from .comparator import Comparator
from .utils.correlation import calculate_average_correlations
from .utils.db_index import calculate_davies_bouldin_index

log = utils.get_pylogger(__name__)


class CorrelationComparator(Comparator):
    def __init__(self, num_samples_per_class: int):
        super().__init__()
        self.num_samples_per_class = num_samples_per_class

    def compare(
        self,
        models: Dict[str, List[LightningModule]],
        datamodule: LightningDataModule,
        output_path: Path,
    ):

        results = {
            label: self.sample_model_metrics(model, datamodule) for label, model in models.items()
        }

        Path(output_path).mkdir(parents=True, exist_ok=True)

        correlations_path = output_path / "correlations.png"
        fig.Correlation(
            f"Hidden layer correlation for {datamodule.name}",
            {label: element["correlations"] for label, element in results.items()},
        ).save(correlations_path)
        log.info(f"New correlation comparison plot saved to {correlations_path}")

        db_index_path = output_path / "db_index.png"
        fig.DaviesBouldinIndex(
            f"DB-Index for {datamodule.name}",
            {label: element["db_index"] for label, element in results.items()},
        ).save(db_index_path)
        log.info(f"New DB Index comparison plot saved to {db_index_path}")

    def sample_model_metrics(
        self, models: List[InspectableModule], datamodule: LightningDataModule
    ) -> Dict[str, Any]:
        activities = np.zeros(
            (
                len(models),
                datamodule.num_classes,
                models[0].net.num_hidden_layers,
                self.num_samples_per_class,
                models[0].net.hidden_layer_width,
            )
        )

        for seed, seed_model in enumerate(
            track(models, description="Sampling metrics for model...")
        ):

            seed_model.eval()
            inspected_model = seed_model.inspect()
            datamodule.setup("test")

            with torch.no_grad():
                for c, class_dataloader in datamodule.test_dataloader_by_class(
                    self.num_samples_per_class
                ).items():
                    samples, _ = next(iter(class_dataloader))
                    _, hidden_layer_activities = inspected_model.forward(samples)
                    activities[seed, c] = np.array(
                        [a.cpu().numpy() for a in hidden_layer_activities]
                    )

        activities = np.array(activities)

        return {
            "activities": activities,
            "correlations": calculate_average_correlations(activities),
            "db_index": calculate_davies_bouldin_index(activities),
        }
