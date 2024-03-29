from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import LightningDataModule
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
        models: Dict[str, Dict[str, tuple[Union[DictConfig, ListConfig], InspectableModule]]],
        datamodule: LightningDataModule,
        output_path: Path,
    ):

        results = {
            label: self.sample_model_metrics(models, datamodule)
            for label, models in models.items()
        }

        Path(output_path).mkdir(parents=True, exist_ok=True)

        config_path = output_path / "config.yaml"

        output_config = DictConfig(
            {
                "comparison": {
                    label: [label for label in models.keys()] for label, models in models.items()
                },
                "model_config": OmegaConf.merge(
                    *[model[0] for models in models.values() for model in models.values()]
                ),
            }
        )

        with open(config_path, "w") as config_file:
            OmegaConf.save(config=output_config, f=config_file.name)

        correlations_path = output_path / "correlations.png"
        fig.Correlation(
            "Hidden layer correlation",
            {label: element["correlations"] for label, element in results.items()},
        ).save(correlations_path)
        log.info(f"New correlation comparison plot saved to {correlations_path}")

        db_index_path = output_path / "db_index.png"
        fig.DaviesBouldinIndex(
            "DB-Index",
            {label: element["db_index"] for label, element in results.items()},
        ).save(db_index_path)
        log.info(f"New DB Index comparison plot saved to {db_index_path}")

    def sample_model_metrics(
        self,
        models: Dict[str, tuple[Union[DictConfig, ListConfig], InspectableModule]],
        datamodule: LightningDataModule,
    ) -> Dict[str, Any]:

        first_model = list(models.values())[0][1]

        activities = np.zeros(
            (
                len(models),
                datamodule.num_classes,
                first_model.net.num_hidden_layers,
                self.num_samples_per_class,
                first_model.net.hidden_layer_width,
            )
        )

        for seed, (config, model) in enumerate(
            track(models.values(), description="Sampling model activities...")
        ):

            model.eval()
            inspected_model = model.inspect()
            datamodule.setup("test")

            with torch.no_grad():
                for c, class_dataloader in datamodule.test_dataloader_by_class(
                    self.num_samples_per_class
                ).items():
                    samples, _ = next(iter(class_dataloader))
                    _, hidden_layer_activities = inspected_model.forward(samples)
                    activities[seed, c] = np.array(
                        [a.reshape(a.size(0), -1).cpu().numpy() for a in hidden_layer_activities]
                    )

        return {
            "activities": activities,
            "correlations": calculate_average_correlations(activities),
            "db_index": calculate_davies_bouldin_index(activities),
        }
