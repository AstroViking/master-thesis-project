import torch
import pandas as pd

from .sigint_handler_context import SigintHandlerContext


def train_model(model: torch.nn.Module, train_dataset: torch.utils.data.Dataset, test_dataset: torch.utils.data.Dataset, num_epochs: int, batch_size: int, criterion, optimizer: torch.optim.Optimizer, device: str) -> pd.DataFrame:

    results = pd.DataFrame(index=[e for e in range(num_epochs)], columns=pd.MultiIndex.from_product([["Train", "Test"], ["Loss", "Accuracy"]]))    
    
    model.to(device).train()

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                    batch_size=batch_size, 
                                    shuffle=True)
    
    
    n_total_steps = len(train_loader)
    test_accuracy = 0.0

    with SigintHandlerContext("Stopping training after end of current epoch...") as context:

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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step() 

                train_accuracy = 100.0 * n_train_correct / n_train_samples

                if (i+1) % 100 == 0:
                    print (f"Epoch [{epoch+1}/{num_epochs}], Step[{i+1}/{n_total_steps}], Train Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
                
            test_accuracy = evaluate_model(model, test_dataset, batch_size, device)
            model.train()

            results.loc[epoch, ("Train", "Loss")] = loss.item()
            results.loc[epoch, ("Train", "Accuracy")] = train_accuracy
            results.loc[epoch, ("Test", "Accuracy")] = test_accuracy

            print (f"Epoch [{epoch+1}/{num_epochs}], Step[{i+1}/{n_total_steps}], Train Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

            if context.sigint_received:
                break
        
    model.eval()

    return results.dropna(how="all")
        


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