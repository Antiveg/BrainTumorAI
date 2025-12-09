import tensorflow as tf
from keras.models import Model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model: Model, test_dataset: tf.data.Dataset, class_names: List[str], save_path=None) -> tuple:

    eval_metrics = model.evaluate(test_dataset)
    
    test_loss = float(eval_metrics[0])
    test_accuracy = float(eval_metrics[1])
    
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    y_true = []
    y_pred = []

    for x_batch, y_batch in test_dataset:
        preds = model.predict(x_batch)
        preds = np.argmax(preds, axis=-1)

        y_batch = np.argmax(y_batch, axis=-1) if len(y_batch.shape) > 1 else y_batch.numpy()
        y_true.extend(y_batch)
        y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names if class_names else None
    )

    print("\n=== Classification Report ===")
    print(report)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    plt.show()

    return test_loss, test_accuracy, report, cm
