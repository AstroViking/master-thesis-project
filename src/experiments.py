from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torchinfo import summary as summarize_model
import torch_optimizer as optim 

from transfer_learning_criticality.neural_nets import FDNN, CNN
import transfer_learning_criticality.figures as fig
from transfer_learning_criticality.util.model import train_model, evaluate_model
from transfer_learning_criticality.util.correlation import calculate_average_correlations

# Select which dataset to use (either "mnist", "fashion-mnist" or "cifar-10")
dataset_identifier = "mnist"

use_pretrained = True
initialize_at_criticality = True
use_cnn = False

# Specify how many samples to use per class to test for correlation
num_samples_per_class = 100

# Specify model parameters

hidden_layer_width = 100 # Layer with for FDNN, number of channels for CNN
num_hidden_layers = 20
num_classes = 10
num_epochs = 20
batch_size = 100
learning_rate = 0.001

model_prefix = f"{'cnn' if use_cnn else 'fdnn'}_{dataset_identifier}_{'critical' if initialize_at_criticality else 'non-critical'}_c-{num_classes}_h-{num_hidden_layers}x{hidden_layer_width}_b-{batch_size}_e{num_epochs}_lr-{learning_rate}"
plot_prefix = f"{num_samples_per_class}-samples"

data_path = Path(__file__).parent.parent / "data"
output_path = Path(__file__).parent.parent / "output" / model_prefix
plot_path = output_path / "plots"

data_path.mkdir(parents=True, exist_ok=True)
output_path.mkdir(parents=True, exist_ok=True)
plot_path.mkdir(parents=True, exist_ok=True)

random_seed = 1234
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Import dataset 
if dataset_identifier == "mnist":
    dataset = torchvision.datasets.MNIST
elif dataset_identifier == "fashion-mnist":
    dataset = torchvision.datasets.FashionMNIST
elif dataset_identifier == "cifar-10":
    dataset = torchvision.datasets.CIFAR10
else:
    print("Invalid dataset identifier specified!")
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
input_size = len(train_dataset[0][0].flatten())
input_width = input_shape[1]

# Initializing the model
if use_cnn:
    num_input_channels = input_shape[0] if len(input_shape) == 3 else 1
    model: torch.nn.Module = CNN(input_width, num_input_channels, hidden_layer_width, num_hidden_layers, num_classes, initialize_at_criticality)
else:

    init_weight_mean = 0
    init_bias_mean = 0
    if initialize_at_criticality:
        init_bias_std = np.sqrt(0.05) # Value obtained using Mean Field criticality theory
        init_weight_std = np.sqrt(1.7603915227624916) # Value obtained using Mean Field criticality theory (1.7603915227624916)
    else:
        init_bias_std = np.sqrt(0.05) 
        init_weight_std = np.sqrt(1)

    model = FDNN(input_size, hidden_layer_width, num_hidden_layers, num_classes, init_weight_mean, init_weight_std, init_bias_mean, init_bias_std)

summarize_model(model, (batch_size, *input_shape))

model_path = output_path / "model.zip"
model_metrics_path = output_path / "model_metrics.pickle"

if use_pretrained and model_path.exists() and model_metrics_path.exists():
    model.load_state_dict(torch.load(model_path))
    model_metrics = pd.read_pickle(model_metrics_path)
    

else:
    optimizer = optim.DiffGrad(model.parameters(), lr=learning_rate) if use_cnn else torch.optim.Adam(model.parameters(), lr=learning_rate)

    model_metrics = train_model(model, train_dataset, test_dataset, num_epochs, batch_size, 
        torch.nn.CrossEntropyLoss(), 
        optimizer,
        device=device
    )

    torch.save(model.state_dict(), model_path)
    model_metrics.to_pickle(model_metrics_path)

fig.model_accuracy_vs_epoch("", model_metrics).write_image(plot_path / "model_accuracy_vs_epoch.png", scale=3)

print(f"Accuracy of the network on test dataset after training: {model_metrics.iloc[-1].loc['Test Accuracy']}")

with torch.no_grad():

    model = model.to(device).eval()

    # Plot variances of weights and bias across layers 
    weight_variances = np.zeros(num_hidden_layers + 1)
    bias_variances = np.zeros(num_hidden_layers + 1)

    if isinstance(model, FDNN):
        i = 0
        for layer in model.modules():
            if isinstance(layer, torch.nn.Linear):                
                weight_variances[i] = np.std(layer.weight.cpu().numpy() * np.sqrt(layer.in_features))**2
                bias_variances[i] = np.std(layer.bias.cpu().numpy())**2
                i += 1
        
        fig.model_weight_bias_variance(f"Weight and bias variance across layers for network trained with {dataset_identifier} dataset", weight_variances, bias_variances).write_image(plot_path / f"{plot_prefix}_weight_bias_variance.png", scale=3)

    hidden_layer_activities_path = output_path / "hidden_layer_activities.pickle"
    
    if use_pretrained and hidden_layer_activities_path.exists():
        hidden_layer_activities = pd.read_pickle(hidden_layer_activities_path)
    
    else:

        # Calculate activities for different classes/samples
        image_classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        hidden_layer_activities = pd.DataFrame(index=pd.MultiIndex.from_product([[c for c in image_classes], [s for s in range(num_samples_per_class)]], names=["Class", "Sample"]), columns=pd.MultiIndex.from_product([[l for l in range(num_hidden_layers)], [n for n in range(model.hidden_layer_width)]], names=["Layer", "Neuron"]))

        for image_class in image_classes:
            class_indices = [idx for idx, target in enumerate(test_dataset.targets) if target == image_class]
            class_subset = torch.utils.data.Subset(test_dataset, class_indices)

            random_class_indices = np.random.choice(len(class_subset), size=num_samples_per_class, replace=False)

            for i in range(num_samples_per_class):
                random_sample = class_subset[random_class_indices[i]][0]
                output_activity, hidden_layer_activity =  model.forward(random_sample.to(device), True)
                hidden_layer_activities.loc[(image_class, i), :] = hidden_layer_activity.flatten()

        
        hidden_layer_activities.to_pickle(hidden_layer_activities_path)

    correlations, combined_correlations = calculate_average_correlations(hidden_layer_activities)

    fig.average_correlation_between_classes(f"Average correlation of input vectors accros {num_samples_per_class} samples from {dataset_identifier}", correlations).write_image((plot_path / f"{plot_prefix}_average_correlation_between_classes.png"), scale=3)
    fig.average_correlation_same_vs_different_class(f"Average correlation of input vectors accros {num_samples_per_class} samples from {dataset_identifier}", combined_correlations).write_image((plot_path / f"{plot_prefix}_average_correlation_same_vs_different_class.png"), scale=3)
    fig.davies_bouldin_index(f"DB index of activity vectors accros {num_samples_per_class} samples from {dataset_identifier}", hidden_layer_activities).write_image((plot_path / f"{plot_prefix}_cluster_db_index.png"), scale=3)    