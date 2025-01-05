import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
import random
import math

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

    # Collect all data for random sampling
    all_images = []
    all_labels = []

    # Extract all images and labels from the data loader
    for images, labels in data_loader:
        all_images.append(images)
        all_labels.append(labels)

    # Flatten the list and stack tensors
    all_images = torch.cat(all_images)
    all_labels = torch.cat(all_labels)

    # Randomly choose indices for visualization
    indices = random.sample(range(len(all_images)), num_images)

    # Determine grid size (rows and columns)
    num_cols = min(5, num_images)  # Limit to 5 columns
    num_rows = math.ceil(num_images / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
    axes = axes.flatten()  # Flatten to easily iterate

    with torch.no_grad():
        # Get selected images and labels
        images = all_images[indices].to(device)
        labels = all_labels[indices].to(device)

        # Predictions
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        for i, (img, label, pred, ax) in enumerate(zip(images, labels, preds, axes)):
            # Unnormalize image for visualization
            img = img.cpu().numpy().transpose((1, 2, 0))  # CHW -> HWC
            img = img * 0.5 + 0.5  # Reverse normalization

            # Plot the image
            ax.imshow(np.clip(img, 0, 1))
            ax.axis('off')
            ax.set_title(f"True: {classes[label.item()]}\nPred: {classes[pred.item()]}", c="green" if classes[label.item()] == classes[pred.item()] else "red")

    plt.tight_layout()
    plt.show()

def show_classification_metrics(model, data_loader, class_names, device):
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
    
    # Calculate metrics
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Display metrics
    metrics = {
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Accuracy': accuracy
    }
    
    # plt.figure(figsize=(8, 6))
    # plt.bar(metrics.keys(), metrics.values(), color=['skyblue', 'orange', 'green', 'red'])
    # plt.title('Classification Metrics')
    # plt.ylabel('Score')
    # plt.ylim(0, 1)
    # plt.show()

    # Print metrics in text form
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
