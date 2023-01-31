from pathlib import Path
import yaml
import click
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torchinfo import summary as summarize_model
import torch_optimizer as optim 

from transfer_learning_criticality.schemas.config import config_schema
from transfer_learning_criticality.neural_nets import FeedForwardNet, ConvolutionalNet
import transfer_learning_criticality.figures as fig
from transfer_learning_criticality.util.model import train_model
from transfer_learning_criticality.util.correlation import calculate_average_correlations
from transfer_learning_criticality.util.mean_field import MeanField

from schema import SchemaError

@click.group()
@click.option("-c", "--configpath", default=Path(__file__).parent.parent / "config.yaml", type=click.Path(exists=True),  help="Path to the config file.")
@click.option("-t", "--train", default=False, help="Train model even if pretrained model exists.")
@click.pass_context
def cli(ctx, configpath, train):
    with open(configpath, "r") as configfile:
        config = yaml.safe_load(configfile)
        try:
            config_schema.validate(config)
        except SchemaError as error:
            print(f"Specified config is not valid. Validation error was {error}")
            exit(0)


        ctx.ensure_object(dict)
        ctx.obj["config"] = config
        ctx.obj["train"] = train


@cli.command()
@click.pass_context
def run_all(ctx):
    ctx.ensure_object(dict)
    n_experiments = len(ctx.obj["config"]["experiments"])

    for i in range(n_experiments):
        run(ctx, i)


@cli.command()
@click.argument("experiment_index", default=0)
@click.option("-s", "--show-summary", "show_model_summary", is_flag=True, default=False, help="Print model summary")
@click.pass_context
def run(ctx, experiment_index, show_model_summary):
    
    ctx.ensure_object(dict)
    config = ctx.obj["config"]["experiments"][experiment_index]

    print(f"Running experiment with config {config}")

    use_pretrained = ctx.obj["train"]
    initialize_at_criticality = True

    experiment_foldername = f"{config['model']}_{config['dataset']}_{'critical' if initialize_at_criticality else 'non-critical'}_h-{config['model_parameters']['num_hidden_layers']}x{config['model_parameters']['hidden_layer_width']}_b-{config['batch_size']}_e{config['num_epochs']}_lr-{config['learning_rate']}"

    data_path = Path(__file__).parent.parent / "data"
    output_path = Path(__file__).parent.parent / "output" / experiment_foldername
    plot_path = output_path / "plots"

    data_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / "model.zip"
    model_metrics_path = output_path / "model_metrics.pickle"

    random_seed = 1234
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Import dataset 
    if config["dataset"] == "MNIST":
        dataset = torchvision.datasets.MNIST
    elif config["dataset"] == "FashionMNIST":
        dataset = torchvision.datasets.FashionMNIST
    elif config["dataset"] == "CIFAR10":
        dataset = torchvision.datasets.CIFAR10
    else:
        print(f"Dataset {config['dataset']} is not valid!")
        exit(1)

    # Load dataset
    train_dataset = dataset(root=data_path, 
                            train=True, 
                            transform=transforms.ToTensor(),  
                            download=True)
    test_dataset = dataset(root=data_path, 
                            train=False, 
                            transform=transforms.ToTensor())

    # Set the device
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu" # type: ignore[attr-defined]

    input_shape = train_dataset[0][0].shape
    num_classes = len(train_dataset.classes)

    if config["model_parameters"]["non_linearity"] == "tanh":
        mean_field_calculator = MeanField(np.tanh, lambda x: 1./ np.cosh(x)**2)
        non_linearity = torch.nn.Tanh()

    elif config["model_parameters"]["non_linearity"] == "selu":
        ALPHA = 1.6732632423543772848170429916717
        LAMBDA = 1.0507009873554804934193349852946
        mean_field_calculator = MeanField(lambda x: x <= LAMBDA * (ALPHA * np.exp(x) - ALPHA) if x <= 0.0 else LAMBDA * x, lambda x: LAMBDA * ALPHA * np.exp(x) if x <= 0.0 else LAMBDA)
        non_linearity = torch.nn.SELU()


    if initialize_at_criticality:
        qstar = 1./config["model_parameters"]["num_hidden_layers"]
    else:
        qstar = 1

    init_weight_mean = 0
    init_bias_mean = 0
    init_weight_var, init_bias_var = mean_field_calculator.sw_sb(qstar, 1)    

    if config["model"] == "FeedForwardNet":
        model = FeedForwardNet(
            input_shape, 
            config["model_parameters"]["hidden_layer_width"], 
            config["model_parameters"]["num_hidden_layers"], 
            num_classes, 
            init_weight_mean, 
            init_weight_var, 
            init_bias_mean, 
            init_bias_var,
            non_linearity
        )

    elif config["model"] == "ConvolutionalNet":
        model = ConvolutionalNet(
            input_shape,
            config["model_parameters"]["hidden_layer_width"],
            config["model_parameters"]["num_hidden_layers"],
            num_classes, 
            init_weight_mean, 
            init_weight_var, 
            init_bias_mean, 
            init_bias_var,
            non_linearity
        )

    else: 
        print(f"Model type {config['model']} is not valid!")
        exit(1)

    if show_model_summary:
        summarize_model(model, (config["batch_size"], *input_shape))


    if use_pretrained and model_path.exists() and model_metrics_path.exists():
        model.load_state_dict(torch.load(model_path))
        model_metrics = pd.read_pickle(model_metrics_path)

    else:
        model_metrics = train_model(model, train_dataset, test_dataset, config["num_epochs"], config["batch_size"], 
            torch.nn.CrossEntropyLoss(), 
            optim.DiffGrad(model.parameters(), lr=config["learning_rate"]) if isinstance(model, ConvolutionalNet) else torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.95),
            device=device
        )

        torch.save(model.state_dict(), model_path)
        model_metrics.to_pickle(model_metrics_path)

    fig.model_accuracy_vs_epoch("", model_metrics).write_image(plot_path / "model_accuracy_vs_epoch.png", scale=3)

    print(f"Accuracy of the network on test dataset after training: {model_metrics.iloc[-1].loc[('Test', 'Accuracy')]}")

    with torch.no_grad():

        model = model.to(device).eval()

        # Plot variances of weights and bias across layers 
        if isinstance(model, FeedForwardNet):
            
            weight_variances = np.zeros(config["model_parameters"]["num_hidden_layers"] + 1)
            bias_variances = np.zeros(config["model_parameters"]["num_hidden_layers"] + 1)
            
            i = 0
            for layer in model.modules():
                if isinstance(layer, torch.nn.Linear):                
                    weight_variances[i] = np.var(layer.weight.cpu().numpy() * np.sqrt(layer.in_features))
                    bias_variances[i] = np.var(layer.bias.cpu().numpy())
                    i += 1

        elif isinstance(model, ConvolutionalNet):

            weight_variances = np.zeros(config["model_parameters"]["num_hidden_layers"] + 1 + 1)
            bias_variances = np.zeros(config["model_parameters"]["num_hidden_layers"] + 1 + 1)

            i = 0
            for layer in model.modules():
                if isinstance(layer, torch.nn.Conv2d):                
                    weight_variances[i] = np.var(layer.weight.cpu().numpy() * np.sqrt(layer.in_channels * np.prod(layer.kernel_size)))
                    
                    if layer.bias is not None:
                        bias_variances[i] = np.var(layer.bias.cpu().numpy())
                    i += 1
            
        fig.model_weight_bias_variance(f"Weight and bias variance across layers for network trained with {config['dataset']} dataset", weight_variances, bias_variances).write_image(plot_path / f"model_weight_bias_variance.png", scale=3)

        hidden_layer_activities_path = output_path / "hidden_layer_activities.pickle"
        
        if use_pretrained and hidden_layer_activities_path.exists():
            hidden_layer_activities = pd.read_pickle(hidden_layer_activities_path)
        
        else:

            input_classes = range(len(test_dataset.classes))

            hidden_layer_activities = pd.DataFrame(index=pd.MultiIndex.from_product([[c for c in input_classes], [s for s in range(config["num_samples_per_class"])]], names=["Class", "Sample"]), columns=pd.MultiIndex.from_product([[l for l in range(config["model_parameters"]["num_hidden_layers"])], [n for n in range(config["model_parameters"]["hidden_layer_width"])]], names=["Layer", "Neuron"]))

            for c in input_classes:
                class_indices = [idx for idx, target in enumerate(test_dataset.targets) if target == c]
                class_subset = torch.utils.data.Subset(test_dataset, class_indices)

                random_class_indices = np.random.choice(len(class_subset), size=config["num_samples_per_class"], replace=False)

                for i in range(config["num_samples_per_class"]):
                    random_sample = class_subset[random_class_indices[i]][0]
                    output_activity, hidden_layer_activity =  model.forward(random_sample.to(device), True)
                    hidden_layer_activities.loc[(c, i), :] = hidden_layer_activity.flatten()

            
            hidden_layer_activities.to_pickle(hidden_layer_activities_path)

        correlations, combined_correlations = calculate_average_correlations(hidden_layer_activities)

        samples_plot_suffix = f"{config['num_samples_per_class']}-samples"

        fig.average_correlation_between_classes(f"Average correlation of input vectors accros {config['num_samples_per_class']} samples from {config['dataset']}", correlations).write_image((plot_path / f"average_correlation_between_classes-{samples_plot_suffix}.png"), scale=3)
        fig.average_correlation_same_vs_different_class(f"Average correlation of input vectors accros {config['num_samples_per_class']} samples from {config['dataset']}", combined_correlations).write_image((plot_path / f"average_correlation_same_vs_different_class-{samples_plot_suffix}.png"), scale=3)
        fig.davies_bouldin_index(f"DB index of activity vectors accros {config['num_samples_per_class']} samples from {config['dataset']}", hidden_layer_activities).write_image((plot_path / f"cluster_db_index-{samples_plot_suffix}.png"), scale=3)


if __name__ == "__main__":
    cli()