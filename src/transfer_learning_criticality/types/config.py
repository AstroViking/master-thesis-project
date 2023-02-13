from dataclass_wizard import YAMLWizard
from dataclasses import dataclass

@dataclass
class ModelParameters:
    hidden_layer_width: int
    num_hidden_layers: int
    non_linearity: str

@dataclass
class Model:
    type: str
    parameters: ModelParameters

@dataclass
class TrainParameters:
    dataset: str
    num_epochs: int
    batch_size: int
    learning_rate: float

@dataclass
class TrainPhases:
    initial: TrainParameters
    transfer: TrainParameters

@dataclass
class EvaluationParameters:
    num_samples_per_class: int

@dataclass
class Experiment:
    model: Model
    training: TrainPhases
    evaluation: EvaluationParameters

@dataclass
class Config(YAMLWizard):
    experiments: list[Experiment]