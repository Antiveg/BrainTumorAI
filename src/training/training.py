from keras.applications import ResNet50, VGG16, MobileNetV2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import keras
from tensorflow.data import Dataset
import os

def train_model(
    model: keras.Model, 
    train_dataset: Dataset, 
    val_dataset: Dataset, 
    batch_size: int = 32, 
    epochs: int = 10
) -> Tuple[dict, keras.Model]:
    
    callbacks = []

    early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    callbacks.append(early_stopping)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, mode='min')
    callbacks.append(reduce_lr)

    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        batch_size=batch_size
    )

    return history.history, model

def plot_training_history(history: dict, model_name: str = 'Model', save_path: Optional[str] = None) -> None:

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training accuracy')
    plt.plot(history['val_accuracy'], label='Validation accuracy')
    plt.title(f"{model_name}'s Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training loss')
    plt.plot(history['val_loss'], label='Validation loss')
    plt.title(f"{model_name}'s Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()
    plt.close()