from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torchinfo import summary as summarize_model

from transfer_learning_criticality.neural_nets import FDNN
import transfer_learning_criticality.figures as fig
from transfer_learning_criticality.util.model import train_model, evaluate_model


# Select which dataset to use (either "mnist" or "fashion-mnist")
dataset_identifier = "fashion-mnist"

train = True
use_pretrained = True
initialize_at_criticality = True

# Specify how many samples to use per class to test for correlation
num_samples_per_class = 100

# Specify model parameters
init_weight_mean = 0
init_bias_mean = 0
hidden_layer_width = 100
num_hidden_layers = 20
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

if initialize_at_criticality:
    init_bias_std = np.sqrt(0.05) # Value obtained using Mean Field criticality theory
    init_weight_std = np.sqrt(1.7603915227624916) # Value obtained using Mean Field criticality theory (1.7603915227624916)
else:
    init_bias_std = np.sqrt(0.05) 
    init_weight_std = np.sqrt(1)

experiment_foldername = f"{dataset_identifier}_{'critical' if initialize_at_criticality else 'non-critical'}_{num_samples_per_class}-samples_{'trained' if train or use_pretrained else 'untrained'}"
model_filename = f"{dataset_identifier}_{'critical' if initialize_at_criticality else 'non-critical'}_c-{num_classes}_h-{num_hidden_layers}x{hidden_layer_width}_b-{batch_size}_e{num_epochs}_lr-{learning_rate}.zip"

data_path = Path(__file__).parent.parent / "data"
output_path = Path(__file__).parent.parent / "output"
model_path = output_path / "pretrained_models" / model_filename
plot_path = output_path / "plots" / experiment_foldername

model_path.parent.mkdir(parents=True, exist_ok=True)
plot_path.mkdir(parents=True, exist_ok=True)

random_seed = 1234
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Import dataset 
if dataset_identifier == "mnist":
    train_dataset = torchvision.datasets.MNIST(root=data_path, 
                                            train=True, 
                                            transform=transforms.ToTensor(),  
                                            download=True)
    test_dataset = torchvision.datasets.MNIST(root=data_path, 
                                            train=False, 
                                            transform=transforms.ToTensor())
else:
    train_dataset = torchvision.datasets.FashionMNIST(root=data_path, 
                                            train=True, 
                                            transform=transforms.ToTensor(),  
                                            download=True)
    test_dataset = torchvision.datasets.FashionMNIST(root=data_path, 
                                            train=False, 
                                            transform=transforms.ToTensor())

# Set the device
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.backends.cuda.is_available() else "cpu" # type: ignore[attr-defined]

input_size = len(train_dataset[0][0].flatten())

# Initializing the model
model = FDNN(input_size, hidden_layer_width, num_hidden_layers, num_classes, init_weight_mean, init_weight_std, init_bias_mean, init_bias_std).to(device)
model.to(device)

if use_pretrained and (model_path).exists():
    model.load_state_dict(torch.load(model_path))
    model.eval()

elif train:
    train_model(model, train_dataset, num_epochs, batch_size, 
        torch.nn.CrossEntropyLoss(), 
        torch.optim.Adam(model.parameters(), lr=learning_rate),
        device=device)

    torch.save(model.state_dict(), model_path)

summarize_model(model, (batch_size, input_size))
evaluate_model(model.to(device), test_dataset, batch_size, device)

with torch.no_grad():

    # Plot variances of weights and bias across layers 
    weight_variances = np.zeros(num_hidden_layers + 1)
    bias_variances = np.zeros(num_hidden_layers + 1)
    
    i = 0
    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear):                
            weight_variances[i] = np.std(layer.weight.cpu().numpy() * np.sqrt(layer.in_features))**2
            bias_variances[i] = np.std(layer.bias.cpu().numpy())**2
            i += 1

    fig.layer_variance(f"Weight and bias variance across layers for network trained with {dataset_identifier} dataset", weight_variances, bias_variances).write_image(plot_path / f"weight_bias_variance.png")

    # Calculate activities for different classes/samples
    image_classes = [1, 2, 3, 9]
    hidden_layer_activities = pd.DataFrame(index=pd.MultiIndex.from_product([[c for c in image_classes], [s for s in range(num_samples_per_class)]], names=["Class", "Sample"]), columns=pd.MultiIndex.from_product([[l for l in range(num_hidden_layers)], [n for n in range(hidden_layer_width)]], names=["Layer", "Neuron"]))

    for image_class in image_classes:
        class_indices = [idx for idx, target in enumerate(test_dataset.targets) if target == image_class]
        class_subset = torch.utils.data.Subset(test_dataset, class_indices)

        random_class_indices = np.random.choice(len(class_subset), size=num_samples_per_class, replace=False)

        for i in range(num_samples_per_class):
            output_activity, hidden_layer_activity =  model.forward(class_subset[random_class_indices[i]][0].reshape(input_size).to(device), True)
            hidden_layer_activities.loc[(image_class, i), :] = hidden_layer_activity.flatten()

    fig.average_correlation(f"Average correlation of input vectors accros {num_samples_per_class} samples from {dataset_identifier}", hidden_layer_activities).write_image((plot_path / "average_class_correlations.png"))
    fig.davies_bouldin_index(f"DB index of activity vectors accros {num_samples_per_class} samples from {dataset_identifier}", hidden_layer_activities).write_image((plot_path / "cluster_db_index.png"))    
