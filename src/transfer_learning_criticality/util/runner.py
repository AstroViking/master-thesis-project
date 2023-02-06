from pprint import pprint
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torchinfo import summary as summarize_model
import torch_optimizer as optim 

from ..neural_nets import FeedForwardNet, ConvolutionalNet
from .. import figures as fig
from ..util.model import train_model
from ..util.correlation import calculate_average_correlations
from ..util.mean_field import MeanField

def run_experiment(config: dict, root_path: Path, use_pretrained: bool=True, show_model_summary=False):

    random_seed = 1234
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    experiment_foldername = f"{config['model']}_{config['dataset']}_h-{config['model_parameters']['num_hidden_layers']}x{config['model_parameters']['hidden_layer_width']}_b-{config['batch_size']}_e{config['num_epochs']}_lr-{config['learning_rate']}"

    data_path = Path(root_path).parent.parent / "data"
    output_path = Path(root_path).parent.parent / "output" / experiment_foldername
    models_path = output_path / "models"
    results_path = output_path / "results"
    plots_path = output_path / "plots"

    print(f"Running experiment with config:")
    pprint(config)
    print(f"Output path is set to: {output_path}\n")

    data_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)
    plots_path.mkdir(parents=True, exist_ok=True)

    critical_weight_var, critical_bias_var = calculate_critical_initialization(config["model_parameters"]["non_linearity"], config["model_parameters"]["num_hidden_layers"])

    initializations = {
        "critical": {
            "bias_mean": 0,
            "bias_var": critical_bias_var,
            "weight_mean": 0,
            "weight_var": critical_weight_var
        },
        "ordered": {
            "bias_mean": 0,
            "bias_var": critical_bias_var * 1.25,
            "weight_mean": 0,
            "weight_var": critical_weight_var * 0.75
        },
        "chaotic": {
            "bias_mean": 0,
            "bias_var": critical_bias_var * 0.25,
            "weight_mean": 0,
            "weight_var": critical_weight_var * 0.75
        },
    }

    metrics_dict = {}
    activities_dict = {}
    correlations_dict = {}

    train_dataset, test_dataset = load_datasets(config["dataset"], data_path)

    for label, initialization in initializations.items():

        model_identifier = f"{label}_{initialization['bias_mean']}_{initialization['bias_var']}_{initialization['weight_mean']}_{initialization['weight_var']}"
        model_path = models_path / f"{model_identifier}.zip"
        model_metrics_path = results_path / f"model_metrics_{model_identifier}.pickle"
        model_newly_trained = False

        model = initialize_network(config["model"], train_dataset[0][0].shape, config["model_parameters"]["hidden_layer_width"], config["model_parameters"]["num_hidden_layers"], len(train_dataset.classes), config["model_parameters"]["non_linearity"], initialization)

        if show_model_summary:
            summarize_model(model, (1, *train_dataset[0][0].shape))

        if use_pretrained and model_path.exists() and model_metrics_path.exists():

            print("Pretrained network found, loading it...")

            model.load_state_dict(torch.load(model_path))
            model_metrics = pd.read_pickle(model_metrics_path)

        else:
            model_newly_trained = True

            model_metrics = train_model(model, train_dataset, test_dataset, config["num_epochs"], config["batch_size"], 
                torch.nn.CrossEntropyLoss(), 
                optim.DiffGrad(model.parameters(), lr=config["learning_rate"]) if isinstance(model, ConvolutionalNet) else torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.95),
                device=device
            )

            torch.save(model.state_dict(), model_path)
            model_metrics.to_pickle(model_metrics_path)

            print(f"Train network with {label} initialization...")
        
        print(f"Accuracy of the network on test dataset after training: {model_metrics.iloc[-1].loc[('Test', 'Accuracy')]}")

        metrics_dict[label] = model_metrics

        num_samples_suffix = f"{config['num_samples_per_class']}-samples"

        model = model.to(device).eval()

        with torch.no_grad():
        
            fig.model_weight_bias_variance(f"Weight and bias variance across layers for network trained with {config['dataset']} dataset", *get_weight_bias_variances(model)).write_image(plots_path / f"{model_identifier}_model_weight_bias_variance.png", scale=3)

            hidden_layer_activities_path = results_path / f"{model_identifier}_hidden_layer_activities_{num_samples_suffix}.pickle"
            
            if not model_newly_trained and hidden_layer_activities_path.exists():
                hidden_layer_activities = pd.read_pickle(hidden_layer_activities_path)
            
            else:
                hidden_layer_activities = sample_hidden_layer_activities(model, test_dataset, config["num_samples_per_class"], config["model_parameters"]["num_hidden_layers"], device)
                hidden_layer_activities.to_pickle(hidden_layer_activities_path)
            
            activities_dict[label] = hidden_layer_activities

            correlations_path = results_path / f"{model_identifier}_correlations_{num_samples_suffix}.pickle"

            if not model_newly_trained and correlations_path.exists():
                correlations = pd.read_pickle(correlations_path)
            else:
                _, correlations = calculate_average_correlations(hidden_layer_activities)
                correlations.to_pickle(correlations_path)

            correlations_dict[label] = correlations
    

    fig.model_accuracy_vs_epoch("Model accuracy", metrics_dict).write_image(plots_path / "accuracy_vs_epoch.png", scale=3)
    fig.davies_bouldin_index(f"DB index of activity vectors accros {config['num_samples_per_class']} samples from {config['dataset']}", activities_dict).write_image((plots_path / f"cluster_db_index-{num_samples_suffix}.png"), scale=3)
    fig.average_correlation_same_vs_different_class(f"Average correlation of input vectors accros {config['num_samples_per_class']} samples from {config['dataset']}", correlations_dict).write_image((plots_path / f"average_correlation_same_vs_different_class-{num_samples_suffix}.png"), scale=3)

    print("Experiment successfully finished!")
    print(f"Output path was set to: {output_path}\n")


def load_datasets(dataset_identifier: str, data_path: Path):
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


def initialize_network(model_identifier: str, input_shape: tuple[int, int, int], hidden_layer_width: int, num_hidden_layers: int, num_classes: int, non_linearity: str, initialization: dict[str, float]):
    if non_linearity== "tanh":
        non_linearity_module: torch.nn.Module = torch.nn.Tanh()

    elif non_linearity == "selu":
        non_linearity_module = torch.nn.SELU()

    if model_identifier == "FeedForwardNet":
        model: torch.nn.Module = FeedForwardNet(
            input_shape, 
            hidden_layer_width, 
            num_hidden_layers, 
            num_classes, 
            initialization["weight_mean"], 
            initialization["weight_var"], 
            initialization["bias_mean"],
            initialization["bias_var"],
            non_linearity_module
        )

    elif model_identifier == "ConvolutionalNet":
        model = ConvolutionalNet(
            input_shape,
            hidden_layer_width, 
            num_hidden_layers,
            num_classes, 
            initialization["weight_mean"], 
            initialization["weight_var"],
            initialization["bias_mean"],
            initialization["bias_var"],
            non_linearity_module
        )

    else: 
        print(f"Model type {model_identifier} is not valid!")
        exit(1)

    return model


def get_weight_bias_variances(model):

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


def sample_hidden_layer_activities(model: torch.nn.Module, test_dataset, num_samples_per_class: int, num_hidden_layers: int, device: str):

    input_classes = range(len(test_dataset.classes))

    hidden_layer_activities = pd.DataFrame(index=pd.MultiIndex.from_product([[c for c in input_classes], [s for s in range(num_samples_per_class)]], names=["Class", "Sample"]), columns=pd.MultiIndex.from_product([[l for l in range(num_hidden_layers)], [n for n in range(model.hidden_layer_width)]], names=["Layer", "Neuron"]))

    for c in input_classes:
        class_indices = [idx for idx, target in enumerate(test_dataset.targets) if target == c]
        class_subset = torch.utils.data.Subset(test_dataset, class_indices)

        random_class_indices = np.random.choice(len(class_subset), size=num_samples_per_class, replace=False)

        for i in range(num_samples_per_class):
            random_sample = class_subset[random_class_indices[i]][0]
            output_activity, hidden_layer_activity =  model.forward(random_sample.to(device), True)
            hidden_layer_activities.loc[(c, i), :] = hidden_layer_activity.flatten()
    
    return hidden_layer_activities