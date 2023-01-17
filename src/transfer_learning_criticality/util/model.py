import torch

def train_model(model: torch.nn.Module, train_dataset: torch.utils.data.Dataset, num_epochs: int, batch_size: int, criterion, optimizer: torch.optim.Optimizer, device: str):
        
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                    batch_size=batch_size, 
                                    shuffle=True)
    
    input_size = len(train_dataset[0][0].flatten())
    
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

def evaluate_model(model, test_dataset, batch_size, device):

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False)
    
    input_size = len(test_dataset[0][0].flatten())

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