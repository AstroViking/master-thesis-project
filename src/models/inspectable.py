from abc import abstractmethod

from pytorch_lightning import LightningModule
from surgeon_pytorch import Inspect


class InspectableModule(LightningModule):
    @abstractmethod
    def inspect(self) -> Inspect:
        raise NotImplementedError("Must override inspect")
