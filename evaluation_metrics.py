import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns


def plot_loss_curves(results):
    # def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """

    # Get the loss values of the results dictionary (training and test)
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    # Figure out how many epochs there were
    epochs = range(len(results["train_loss"]))

    # Setup a plot
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

def evaluate_with_confusion_matrix(model, data_loader, class_names, device):
    model.eval()  # Set model to evaluation mode
    
    all_preds = []
    all_labels = []

    # Collect all predictions and labels
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix as heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

def visualize_predictions(model, data_loader, classes, device, num_images=10):
    model.eval()  # Set the model to evaluation mode
    images_shown = 0

    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    axes = axes.flatten()  # Flatten the axes array for easier iteration

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            # Predictions
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            # Loop through the batch
            for img, label, pred, ax in zip(images, labels, preds, axes):
                if images_shown >= num_images:
                    break
                
                # Unnormalize image for visualization
                img = img.cpu().numpy().transpose((1, 2, 0))  # CHW -> HWC
                img = img * 0.5 + 0.5  # Reverse normalization

                # Plot the image
                ax.imshow(np.clip(img, 0, 1))
                ax.axis('off')
                ax.set_title(f"True: {classes[label.item()]}\nPred: {classes[pred.item()]}")
                
                images_shown += 1

            if images_shown >= num_images:
                break
    
    plt.tight_layout()
    plt.show()
