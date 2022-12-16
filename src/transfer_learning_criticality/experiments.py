import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from neural_nets import SimpleDNN
from noise import GaussianNoiseGenerator

# Set the device
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.backends.cuda.is_available() else "cpu"

model_path = '../../pretrained_model.zip'
train = True
use_pretrained = False

# Specify model parameters
init_weight_mean = 0
init_weight_std = 1 #np.sqrt(1.3)
init_bias_mean = 0
init_bias_std = 1 #np.sqrt(0.01)

input_size = 784 # 28x28
hidden_size = 500
num_hidden_layers = 12
num_classes = 10
num_epochs = 10
batch_size = 100
learning_rate = 0.001

# Import MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                        train=True, 
                                    transform=transforms.ToTensor(),  
                                        download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', 
                                        train=False, 
                                        transform=transforms.ToTensor())
 # Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                        batch_size=batch_size, 
                                        shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                        batch_size=batch_size, 
                                        shuffle=False)

# Initializing the model
model = SimpleDNN(input_size, hidden_size, num_hidden_layers, num_classes, init_weight_mean, init_weight_std, init_bias_mean, init_bias_std).to(device)

if train:

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):  
            # origin shape: [100, 1, 28, 28]
            # resized: [100, 784]
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 

            if (i+1) % 100 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Step[{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            
    torch.save(model.state_dict(), model_path)

if(use_pretrained):
    model.load_state_dict(torch.load(model_path))
    model.eval()


with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item() 

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')

    torch.manual_seed(1234)

    # Correlation through network layers with random noise inputs
    noise_generator = GaussianNoiseGenerator(0, 1)
    noise_vector1 = noise_generator.create(input_size)
    noise_vector2 = noise_generator.create(input_size)
    print(torch.corrcoef(torch.stack((noise_vector1, noise_vector2)))[0,1])
    print(f'Correlation through network layers with random noise inputs: {model.forward_correlation(noise_vector1.to(device), noise_vector2.to(device))}')

    figure = plt.figure(figsize=(8, 8))

    figure.add_subplot(1, 3, 1)
    plt.axis("off")    
    plt.imshow(noise_vector1.reshape(28,28), cmap="gray")

    figure.add_subplot(1, 3, 2)
    plt.axis("off")
    plt.imshow(noise_vector2.reshape(28,28), cmap="gray")

    #plt.show()

    # Correlation through network layers with inputs from the same vs different class
    class1_indices = [idx for idx, target in enumerate(test_dataset.targets) if target in [1]]
    class1_subset = torch.utils.data.Subset(test_dataset, class1_indices)

    class2_indices = [idx for idx, target in enumerate(test_dataset.targets) if target in [2]]
    class2_subset = torch.utils.data.Subset(test_dataset, class2_indices)

    sample_idx1 = torch.randint(len(class1_subset), size=(1,)).item()
    sample_idx2 = torch.randint(len(class1_subset), size=(1,)).item()
    sample_idx3 = torch.randint(len(class2_subset), size=(1,)).item()
    image1, label = class1_subset[sample_idx1]
    image2, label = class1_subset[sample_idx2]
    image3, label = class2_subset[sample_idx3]

    print(f'Correlation through network layers with inputs from different classes: {model.forward_correlation(image1.reshape(input_size).to(device), image3.reshape(input_size).to(device))}')
    print(f'Correlation through network layers with inputs from the same class: {model.forward_correlation(image1.reshape(input_size).to(device), image2.reshape(input_size).to(device))}')

    figure = plt.figure(figsize=(8, 8))
    
    figure.add_subplot(1, 3, 1)
    plt.axis("off")    
    plt.imshow(image1.squeeze(), cmap="gray")

    figure.add_subplot(1, 3, 2)
    plt.axis("off")
    plt.imshow(image2.squeeze(), cmap="gray")

    figure.add_subplot(1, 3, 3)
    plt.axis("off")
    plt.imshow(image3.squeeze(), cmap="gray")

    #plt.show()

    
