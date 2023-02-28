from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

from pytorch_lightning import LightningDataModule

from src.models.inspectable import InspectableModule


class Comparator(ABC):
    @abstractmethod
    def compare(
        self,
        models: Dict[str, Dict[str, InspectableModule]],
        datamodule: LightningDataModule,
        output_path: Path,
    ):
        raise NotImplementedError("Must override compare")
