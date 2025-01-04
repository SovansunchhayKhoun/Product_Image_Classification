import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Predict on test data
def predict_and_evaluate(model, test_data):
  if len(gpus) > 0:
    with tf.device('/GPU:0'):
      predictions = model.predict(test_data)
  else:
    predictions = model.predict(test_data)
  y_pred = np.argmax(predictions, axis=1)  # Convert softmax to class index
  y_true = test_data.classes                # True labels from test data
  return y_pred, y_true

# 4. Confusion Matrix
def plot_confusion_matrix(y_true, y_pred):
  cm = confusion_matrix(y_true, y_pred)

  plt.figure(figsize=(10, 8))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
  plt.xlabel('Predicted Labels')
  plt.ylabel('True Labels')
  plt.title('Confusion Matrix')
  plt.show()

def calculate_metrics(y_true, y_pred):
  accuracy = accuracy_score(y_true, y_pred)
  precision = precision_score(y_true, y_pred, average='weighted')
  recall = recall_score(y_true, y_pred, average='weighted')
  f1 = f1_score(y_true, y_pred, average='weighted')

  print(f"Accuracy:  {accuracy:.4f}")
  print(f"Precision: {precision:.4f}")
  print(f"Recall:    {recall:.4f}")
  print(f"F1-Score:  {f1:.4f}")
  return f1, accuracy, precision, recall

def visualize_predictions_grid(test_data, y_pred, num_images=16, rows=4):
    images, labels = next(test_data)  # Fetch a batch of test images
    cols = num_images // rows         # Calculate the number of columns

    plt.figure(figsize=(15, 15))
    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i])
        pred_label = class_labels[y_pred[i]]
        true_label = class_labels[np.argmax(labels[i])]
        color = 'green' if pred_label == true_label else 'red'
        plt.title(f"Pred: {pred_label}\nTrue: {true_label}", color=color)
        plt.axis('off')

    plt.tight_layout()
    plt.show()
