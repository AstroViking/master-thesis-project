from pprint import pprint
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torchinfo import summary as summarize_model
import torch_optimizer as optim
from tqdm import tqdm 

from ..types.config import Experiment, Model, TrainParameters
from ..neural_nets import BaseNet, FeedForwardNet, ConvolutionalNet
from .. import figures as fig
from ..util.model import train_model
from ..util.correlation import calculate_average_correlations
from ..util.mean_field import MeanField

def get_model_parameter_key(config: Model):
    return f"{config.type}_h-{config.parameters.num_hidden_layers}x{config.parameters.hidden_layer_width}_{config.parameters.non_linearity}"

def get_training_parameter_key(config: TrainParameters):
    return  f"{config.num_epochs}_{config.batch_size}_{config.learning_rate}"

def run_experiment(config: Experiment, root_path: Path, show_model_summary=False):

    random_seed = 1
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    data_path = Path(root_path).parent.parent / "data"
    output_path = Path(root_path).parent.parent / "output" / get_model_parameter_key(config.model) / f"{config.training.initial.dataset}->{config.training.transfer.dataset}"
    models_path = output_path / "models"
    results_path = output_path / "results"
    plots_path = output_path / "plots"
    num_samples_suffix = f"{config.evaluation.num_samples_per_class}-samples"

    print(f"Running experiment with config:")
    pprint(config)
    print(f"Output path is set to: {output_path}\n")

    data_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)
    plots_path.mkdir(parents=True, exist_ok=True)

    critical_weight_var, critical_bias_var = calculate_critical_initialization(config.model.parameters.non_linearity, config.model.parameters.num_hidden_layers)

    initializations = {
        "critical": {
            "bias_mean": 0,
            "bias_var": critical_bias_var,
            "weight_mean": 0,
            "weight_var": critical_weight_var
        },
        "ordered": {
            "bias_mean": 0,
            "bias_var": critical_bias_var * 1.1,
            "weight_mean": 0,
            "weight_var": critical_weight_var * 0.9
        },
        "chaotic": {
            "bias_mean": 0,
            "bias_var": critical_bias_var * 0.9,
            "weight_mean": 0,
            "weight_var": critical_weight_var * 1.1
        },
    }

    initial_training_metrics = {}
    initial_training_activities = {}
    initial_training_correlations = {}

    initial_train_dataset, initial_test_dataset = load_datasets(config.training.initial.dataset, data_path)
    transfer_train_dataset, transfer_test_dataset = load_datasets(config.training.transfer.dataset, data_path)

    for initialization_key, initialization in initializations.items():

        model = initialize_model(
            config.model,
            initial_train_dataset[0][0].shape,
            len(initial_train_dataset.classes), 
            initialization
        )

        if show_model_summary:
            summarize_model(model, (1, *initial_train_dataset[0][0].shape))


        model_key = f"{initialization_key}_initial_{get_training_parameter_key(config.training.initial)}"

        initial_training_metrics[initialization_key] = train_or_load_model_weights(
            model,
            models_path / model_key, 
            config.training.initial,
            initial_train_dataset,
            initial_test_dataset,
            device
        )

        model = model.to(device).eval()

        with torch.no_grad():
        
            fig.model_weight_bias_variance(f"Weight and bias variance across layers for network trained with {config.training.initial.dataset} dataset", *get_weight_bias_variances(model)).write_image(plots_path / f"{model_key}_weight_bias_variance.png", scale=3)

            hidden_layer_activities_path = results_path / f"{model_key}_hidden_layer_activities_{num_samples_suffix}.pickle"
            
            if  hidden_layer_activities_path.exists():
                hidden_layer_activities = pd.read_pickle(hidden_layer_activities_path)
            else:
                hidden_layer_activities = sample_hidden_layer_activities(model, initial_train_dataset, config.evaluation.num_samples_per_class, config.model.parameters.num_hidden_layers, device)
                hidden_layer_activities.to_pickle(hidden_layer_activities_path)
            
            initial_training_activities[initialization_key] = hidden_layer_activities

            correlations_path = results_path / f"{model_key}_correlations_{num_samples_suffix}.pickle"
            
            if correlations_path.exists():
                correlations = pd.read_pickle(correlations_path)
            else:
                correlations = calculate_average_correlations(hidden_layer_activities)
                correlations.to_pickle(correlations_path)

            initial_training_correlations[initialization_key] = correlations
        
    fig.model_accuracy_vs_epoch("Model accuracy vs Epoch (Initial Training)", initial_training_metrics).write_image(plots_path / "initial_accuracy_vs_epoch.png", scale=3)
    fig.davies_bouldin_index(f"DB index of activity vectors accros {config.evaluation.num_samples_per_class} samples from {config.training.initial.dataset}", initial_training_activities).write_image((plots_path / f"initial_cluster_db_index-{num_samples_suffix}.png"), scale=3)
    fig.average_correlation_same_vs_different_class(f"Average correlation of input vectors accros {config.evaluation.num_samples_per_class} samples from {config.training.initial.dataset}", initial_training_correlations).write_image((plots_path / f"initial_average_correlation_same_vs_different_class-{num_samples_suffix}.png"), scale=3)

    for initialization_key in initializations:

        transfer_training_metrics = {}
        transfer_training_activities = {}
        transfer_training_correlations = {}

        for l in range(1, 5):
        #for l in range(config.model.parameters.num_hidden_layers):
            
            model = initialize_model(
                config.model,
                initial_train_dataset[0][0].shape,
                len(initial_train_dataset.classes), 
                initialization
            )
            
            model.load_state_dict(torch.load(models_path / f"{initialization_key}_initial_{get_training_parameter_key(config.training.initial)}.zip"))

            model.change_num_classes(len(transfer_train_dataset.classes))

            model.freeze_first_n_hidden_layers(model.num_hidden_layers - l)

            model.init_weight_var, model.init_bias_var = calculate_critical_initialization(config.model.parameters.non_linearity, l)
            model.reinit_last_n_hidden_layers(l)

            model_key = f"{initialization_key}_transfer_{get_training_parameter_key(config.training.transfer)}_freeze_{l}"

            transfer_training_metrics[f"{initialization_key}_{l}"] = train_or_load_model_weights(
                model,
                models_path / model_key, 
                config.training.transfer,
                transfer_train_dataset,
                transfer_test_dataset,
                device
            )

            model = model.to(device).eval()

            with torch.no_grad():
            
                hidden_layer_activities_path = results_path / f"{model_key}_hidden_layer_activities_{num_samples_suffix}.pickle"
                
                if  hidden_layer_activities_path.exists():
                    hidden_layer_activities = pd.read_pickle(hidden_layer_activities_path)
                else:
                    hidden_layer_activities = sample_hidden_layer_activities(model, transfer_train_dataset, config.evaluation.num_samples_per_class, config.model.parameters.num_hidden_layers, device)
                    hidden_layer_activities.to_pickle(hidden_layer_activities_path)
                
                transfer_training_activities[f"{initialization_key}_{l}"] = hidden_layer_activities

                correlations_path = results_path / f"{model_key}_correlations_{num_samples_suffix}.pickle"
                
                if correlations_path.exists():
                    correlations = pd.read_pickle(correlations_path)
                else:
                    correlations = calculate_average_correlations(hidden_layer_activities)
                    correlations.to_pickle(correlations_path)

                transfer_training_correlations[f"{initialization_key}_{l}"] = correlations
    
        fig.model_accuracy_vs_epoch(f"Model accuracy vs Epoch (Transfer Learning) with {initialization_key} initialization", transfer_training_metrics).write_image(plots_path / f"{initialization_key}_transfer_accuracy_vs_epoch.png", scale=3)
        fig.davies_bouldin_index(f"DB index of activity vectors accros {config.evaluation.num_samples_per_class} samples from {config.training.transfer.dataset}", transfer_training_activities).write_image((plots_path / f"{initialization_key}_transfer_cluster_db_index-{num_samples_suffix}.png"), scale=3)
        fig.average_correlation_same_vs_different_class(f"Average correlation of input vectors accros {config.evaluation.num_samples_per_class} samples from {config.training.transfer.dataset}", transfer_training_correlations).write_image((plots_path / f"{initialization_key}_transfer_average_correlation_same_vs_different_class-{num_samples_suffix}.png"), scale=3)

    print("Experiment successfully finished!")
    print(f"Output path was set to: {output_path}\n")


def load_datasets(dataset_identifier: str, data_path: Path):
    if dataset_identifier.startswith("EMNIST"):

        split = dataset_identifier.split("-")[1]
        train_dataset = torchvision.datasets.EMNIST(root=data_path, 
                                train=True,
                                split=split,
                                transform=transforms.ToTensor(),  
                                download=True)
        test_dataset = torchvision.datasets.EMNIST(root=data_path,
                                split=split,
                                train=False, 
                                transform=transforms.ToTensor())
    else:
        if dataset_identifier == "MNIST":
            dataset = torchvision.datasets.MNIST
        elif dataset_identifier == "FashionMNIST":
            dataset = torchvision.datasets.FashionMNIST
        elif dataset_identifier == "CIFAR10":
            dataset = torchvision.datasets.CIFAR10
        else:
            print(f"Dataset {dataset_identifier} is not valid!")
            exit(1)
        
        train_dataset = dataset(root=data_path, 
                                train=True, 
                                transform=transforms.ToTensor(),  
                                download=True)
        test_dataset = dataset(root=data_path, 
                                train=False, 
                                transform=transforms.ToTensor())

    return train_dataset, test_dataset


def calculate_critical_initialization(non_linearity: str, depth):
    if non_linearity == "tanh":
        mean_field_calculator = MeanField(np.tanh, lambda x: 1./ np.cosh(x)**2)

    elif non_linearity == "selu":
        ALPHA = 1.6732632423543772848170429916717
        LAMBDA = 1.0507009873554804934193349852946
        mean_field_calculator = MeanField(lambda x: x <= LAMBDA * (ALPHA * np.exp(x) - ALPHA) if x <= 0.0 else LAMBDA * x, lambda x: LAMBDA * ALPHA * np.exp(x) if x <= 0.0 else LAMBDA)

    qstar = 1./depth
    return mean_field_calculator.sw_sb(qstar, 1)


def initialize_model(config: Model, input_shape: tuple[int, int, int], num_classes: int, initialization: dict[str, float]) -> BaseNet:
    
    non_linearity = torch.nn.Tanh() if config.parameters.non_linearity == "tanh" else torch.nn.SELU()

    if config.type == "FeedForwardNet":
        model: BaseNet = FeedForwardNet(
            input_shape, 
            config.parameters.hidden_layer_width, 
            config.parameters.num_hidden_layers, 
            num_classes, 
            initialization["weight_mean"], 
            initialization["weight_var"], 
            initialization["bias_mean"],
            initialization["bias_var"],
            non_linearity 
        )

    elif config.type == "ConvolutionalNet":
        model = ConvolutionalNet(
            input_shape,
            config.parameters.hidden_layer_width, 
            config.parameters.num_hidden_layers,
            num_classes, 
            initialization["weight_mean"], 
            initialization["weight_var"],
            initialization["bias_mean"],
            initialization["bias_var"],
            non_linearity
        )

    else: 
        print(f"Network type {type} is not valid!")
        exit(1)

    return model



def train_or_load_model_weights(model: BaseNet, path: Path, config: TrainParameters, train_dataset: torch.utils.data.Dataset, test_dataset:torch.utils.data.Dataset, device: str) -> pd.DataFrame:
        
    model_path = Path(f"{path}.zip")
    metrics_path = path / f"{path}.metrics.pickle"

    if model_path.exists() and metrics_path.exists():

        print(f"Matching model found at {model_path}, loading it...")

        model.load_state_dict(torch.load(model_path))
        metrics = pd.read_pickle(metrics_path)

    else:
        print(f"No existing model found at {model_path}. Training model with parameters:")
        pprint(config)

        metrics = train_model(
            model, 
            train_dataset, 
            test_dataset, 
            config.num_epochs, 
            config.batch_size, 
            torch.nn.CrossEntropyLoss(), 
            optim.DiffGrad(model.parameters(), lr=config.learning_rate) if isinstance(model, ConvolutionalNet) else torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.95),
            device=device
        )

        torch.save(model.state_dict(), model_path)
        metrics.to_pickle(metrics_path)

    print(f"Accuracy of model at {model_path}: {metrics.iloc[-1].loc[('Test', 'Accuracy')]}")
    
    return metrics


def get_weight_bias_variances(model) -> tuple[np.ndarray, np.ndarray]:

    weight_variances = np.array([])
    bias_variances = np.array([])

    if isinstance(model, FeedForwardNet):
        for layer in model.modules():
            if isinstance(layer, torch.nn.Linear):
                weight_variances = np.append(weight_variances, np.var(layer.weight.cpu().numpy() * np.sqrt(layer.in_features)))
                bias_variances = np.append(bias_variances, np.var(layer.bias.cpu().numpy()))

    elif isinstance(model, ConvolutionalNet):

        for layer in model.modules():
            if isinstance(layer, torch.nn.Conv2d):                
                weight_variances = np.append(weight_variances, np.var(layer.weight.cpu().numpy() * np.sqrt(layer.in_channels * np.prod(layer.kernel_size))))
                if layer.bias is not None:
                    bias_variances = np.append(bias_variances, np.var(layer.bias.cpu().numpy()))
    
    return weight_variances, bias_variances


def sample_hidden_layer_activities(model: BaseNet, dataset, num_samples_per_class: int, num_hidden_layers: int, device: str) -> pd.DataFrame:
    classes = Counter(dataset.targets.numpy())
    classes = Counter({k: c for k, c in classes.items() if c > 0})

    hidden_layer_activities = pd.DataFrame(index=pd.MultiIndex.from_product([[c for c in classes], [s for s in range(num_samples_per_class)]], names=["Class", "Sample"]), columns=pd.MultiIndex.from_product([[l for l in range(num_hidden_layers)], [n for n in range(model.hidden_layer_width)]], names=["Layer", "Neuron"]))

    input_class_iterator = tqdm(sorted(classes.keys()))

    for c in input_class_iterator:

        input_class_iterator.set_description(f"Sampling hidden activities for class {c}")

        class_indices = [idx for idx, target in enumerate(dataset.targets) if target == c]
        class_subset = torch.utils.data.Subset(dataset, class_indices)

        random_class_indices = np.random.choice(len(class_subset), size=num_samples_per_class, replace=False)

        for i in range(num_samples_per_class):
            random_sample = class_subset[random_class_indices[i]][0]
            output_activity, hidden_layer_activity =  model.forward(random_sample.to(device), True)
            hidden_layer_activities.loc[(c, i), :] = hidden_layer_activity.flatten()
    
    return hidden_layer_activities