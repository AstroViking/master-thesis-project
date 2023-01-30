import torch
import numpy as np
import pandas as pd

def train_model(model: torch.nn.Module, train_dataset: torch.utils.data.Dataset, test_dataset: torch.utils.data.Dataset, num_epochs: int, batch_size: int, criterion, optimizer: torch.optim.Optimizer, device: str) -> pd.DataFrame:

    results = pd.DataFrame(index=[e for e in range(num_epochs)], columns=["Train Loss", "Test"])    
    
    model.to(device).train()

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                    batch_size=batch_size, 
                                    shuffle=True)
    
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):

        n_train_correct = 0
        n_train_samples = 0

        for i, (inputs, labels) in enumerate(train_loader):  
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            n_train_samples += labels.size(0)
            n_train_correct += (predicted == labels).sum().item()
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 

            if (i+1) % 100 == 0:
                print (f"Epoch [{epoch+1}/{num_epochs}], Step[{i+1}/{n_total_steps}], Loss: {loss.item():.4f}")

        results.loc[epoch, "Train Accuracy"] = 100.0 * n_train_correct / n_train_samples 
        results.loc[epoch, "Test Accuracy"] = evaluate_model(model, test_dataset, batch_size, device)
        model.train()
        
    model.eval()

    return results
        


def evaluate_model(model, test_dataset, batch_size, device) -> float:

    model.to(device).eval()

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False)
    with torch.no_grad():
        n_test_correct = 0
        n_test_samples = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            n_test_samples += labels.size(0)
            n_test_correct += (predicted == labels).sum().item()

        accuracy = 100.0 * n_test_correct / n_test_samples
    
    return accuracy