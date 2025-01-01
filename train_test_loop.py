from tqdm.auto import tqdm
import torch
from torch import nn
from typing import Dict, Tuple, Optional
import warnings
from pathlib import Path

class TrainingConfig:
    def __init__(
        self,
        epochs: int = 5,
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 0.001,
        checkpoint_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Automated mixed precision training if scaler is provided
        if scaler is not None:
            with torch.cuda.amp.autocast():
                y_pred = model(X)
                loss = loss_fn(y_pred, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
        # Calculate accuracy
        with torch.no_grad():
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            total_correct += (y_pred_class == y).sum().item()
            total_samples += y.size(0)
            total_loss += loss.item() * y.size(0)  # Weight loss by batch size

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: str,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            # Forward pass
            test_pred = model(X)
            loss = loss_fn(test_pred, y)
            
            # Calculate metrics
            test_pred_labels = test_pred.argmax(dim=1)
            total_correct += (test_pred_labels == y).sum().item()
            total_samples += y.size(0)
            total_loss += loss.item() * y.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_dir: Path,
    filename: str = "checkpoint.pt"
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / filename
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)

def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
) -> Dict[str, list]:
    results = {
        "train_loss": [], "train_acc": [],
        "test_loss": [], "test_acc": []
    }
    
    # Initialize mixed precision training if using CUDA
    scaler = torch.cuda.amp.GradScaler() if config.device == "cuda" else None
    
    # Early stopping setup
    best_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in tqdm(range(config.epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=config.device,
            scaler=scaler
        )
        
        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=config.device
        )
        
        # Update results
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        
        # Print progress
        print(
            f"Epoch: {epoch+1}/{config.epochs} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )
        
        # Save checkpoint if specified
        if config.checkpoint_dir:
            save_checkpoint(
                model, optimizer, epoch, test_loss,
                config.checkpoint_dir,
                f"checkpoint_epoch_{epoch+1}.pt"
            )
        
        # Early stopping check
        if config.early_stopping_patience:
            if test_loss < best_loss - config.early_stopping_min_delta:
                best_loss = test_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= config.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
    
    return results
