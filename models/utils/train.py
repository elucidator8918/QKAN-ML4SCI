import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

def train(kan_model, trainloader, valloader, criterion, optimizer, num_epochs=10):
    kan_model.to(device)
    
    for epoch in range(num_epochs):
        # Train
        kan_model.train()
        running_loss = 0
        running_accuracy = 0
        
        with tqdm(trainloader, desc=f"Epoch {epoch+1}") as pbar:
            for data, labels in pbar:
                data, labels = data.to(device), labels.to(device)
                
                optimizer.zero_grad()
                output = kan_model(data)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                
                accuracy = (output.argmax(dim=1) == labels.argmax(dim=1)).float().mean()
                
                running_loss += loss.item()
                running_accuracy += accuracy.item()
                
                pbar.set_postfix(
                    loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]["lr"]
                )
                
                # Log training metrics to wandb
                wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy.item()})
        
        # Compute average training loss and accuracy
        avg_train_loss = running_loss / len(trainloader)
        avg_train_accuracy = running_accuracy / len(trainloader)
        
        # Validation
        kan_model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for data, labels in valloader:
                data, labels = data.to(device), labels.to(device)
                output = kan_model(data)
                val_loss += criterion(output, labels).item()
                val_accuracy += (output.argmax(dim=1) == labels.argmax(dim=1)).float().mean().item()
        
        val_loss /= len(valloader)
        val_accuracy /= len(valloader)
        
        # Log validation metrics to wandb
        wandb.log(
            {
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        
        print(f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")
    
    wandb.finish()