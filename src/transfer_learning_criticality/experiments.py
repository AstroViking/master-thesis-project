from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torchvision
import torchvision.transforms as transforms

from neural_nets import FDNN
from noise import GaussianNoiseGenerator
from figures import average_correlation_figure, layer_variance_figure


random_seed = 1234
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Set the device
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.backends.cuda.is_available() else "cpu"

data_path = Path(__file__).parent.parent.parent / "data"
model_path = data_path / "pretrained_model.zip"
plot_path = data_path / "plots"

dataset_identifier = "fashion-mnist" #mnist
train = True
use_pretrained = False

# Specify model parameters
init_weight_mean = 0
init_weight_std = np.sqrt(1.7603915227624916) # Value obtained using Mean Field criticality theory
init_bias_mean = 0
init_bias_std = np.sqrt(0.05) # Value obtained using Mean Field criticality theory

hidden_size = 100
num_hidden_layers = 20
num_classes = 10
num_epochs = 10
batch_size = 100
learning_rate = 0.001

# Specify how many samples we test for correlation across layers
num_correlation_samples = 10

# Import dataset 

if dataset_identifier == 'fashion-mnist':
    train_dataset = torchvision.datasets.FashionMNIST(root=data_path, 
                                            train=True, 
                                            transform=transforms.ToTensor(),  
                                            download=True)
    test_dataset = torchvision.datasets.FashionMNIST(root=data_path, 
                                            train=False, 
                                            transform=transforms.ToTensor())
else:

    train_dataset = torchvision.datasets.MNIST(root=data_path, 
                                            train=True, 
                                            transform=transforms.ToTensor(),  
                                            download=True)
    test_dataset = torchvision.datasets.MNIST(root=data_path, 
                                            train=False, 
                                            transform=transforms.ToTensor())
 # Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                        batch_size=batch_size, 
                                        shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                        batch_size=batch_size, 
                                        shuffle=False)

input_size = len(train_dataset[0][0].flatten())

# Initializing the model
model = FDNN(input_size, hidden_size, num_hidden_layers, num_classes, init_weight_mean, init_weight_std, init_bias_mean, init_bias_std).to(device)

if train:

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):  
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 

            if (i+1) % 100 == 0:
                print (f"Epoch [{epoch+1}/{num_epochs}], Step[{i+1}/{n_total_steps}], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), model_path)

    with torch.no_grad():
        n_test_correct = 0
        n_test_samples = 0
        for images, labels in test_loader:
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            n_test_samples += labels.size(0)
            n_test_correct += (predicted == labels).sum().item() 

        acc = 100.0 * n_test_correct / n_test_samples
        print(f"Accuracy of the network on test dataset: {acc} %")

if(use_pretrained):
    model.load_state_dict(torch.load(model_path))
    model.eval()

with torch.no_grad():

    # Plot variances of weights and bias across layers 
    weight_variances, bias_variances = model.weight_bias_variances()
    layer_variance_figure(f"Weight and bias variance across layers for network trained with {dataset_identifier} dataset", weight_variances, bias_variances).write_image(plot_path / f"{dataset_identifier}_{'trained' if train else 'untrained'}_variance.png")

    # Plot  correlation across layers
    correlations = pd.DataFrame(index=pd.MultiIndex.from_product([["Low Noise", "High Noise", "Same Class", "Different Class"], [l for l in range(num_correlation_samples)]], names=["Category", "Sample"]), columns=[l for l in range(num_hidden_layers + 3)])

    low_noise_generator = GaussianNoiseGenerator(0, 0.1)
    low_noise_vector = low_noise_generator.create(input_size)

    high_noise_generator = GaussianNoiseGenerator(0, 1)
    high_noise_vector = high_noise_generator.create(input_size)

    class1_indices = [idx for idx, target in enumerate(test_dataset.targets) if target in [1]]
    class1_subset = torch.utils.data.Subset(test_dataset, class1_indices)

    class2_indices = [idx for idx, target in enumerate(test_dataset.targets) if target in [2]]
    class2_subset = torch.utils.data.Subset(test_dataset, class2_indices)

    for i in range(num_correlation_samples):
        correlations.loc["Low Noise", i] = model.forward_correlations(low_noise_vector.to(device), low_noise_generator.create(input_size).to(device))
        correlations.loc["High Noise", i] = model.forward_correlations(high_noise_vector.to(device), high_noise_generator.create(input_size).to(device))

        random_class1_indexes = np.random.choice(len(class1_subset), size=3, replace=False)
        random_class2_index = np.random.choice(len(class2_subset))

        image1 = class1_subset[random_class1_indexes[0]][0]
        image2 = class1_subset[random_class1_indexes[1]][0]
        image3 = class2_subset[random_class2_index][0]

        correlations.loc["Same Class", i] = model.forward_correlations(image1.reshape(input_size).to(device), image2.reshape(input_size).to(device))
        correlations.loc["Different Class", i] = model.forward_correlations(image1.reshape(input_size).to(device), image3.reshape(input_size).to(device))

    
    average_correlation_figure(f"Average correlation of input vectors accros {num_correlation_samples} samples from {dataset_identifier}", correlations).write_image((plot_path / f"{dataset_identifier}_{'trained' if train else 'untrained'}_correlation.png"))