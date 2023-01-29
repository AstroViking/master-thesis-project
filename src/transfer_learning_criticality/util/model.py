import torch

def train_model(model: torch.nn.Module, train_dataset: torch.utils.data.Dataset, num_epochs: int, batch_size: int, criterion, optimizer: torch.optim.Optimizer, device: str):
        
    model = model.to(device)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                    batch_size=batch_size, 
                                    shuffle=True)
    
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):  
            images = images.to(device)
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

    model = model.to(device)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False)
    with torch.no_grad():
        n_test_correct = 0
        n_test_samples = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            n_test_samples += labels.size(0)
            n_test_correct += (predicted == labels).sum().item() 

        acc = 100.0 * n_test_correct / n_test_samples
        print(f"Accuracy of the network on test dataset: {acc} %")