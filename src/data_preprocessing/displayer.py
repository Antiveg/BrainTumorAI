import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import Optional

def display_images(dataset, num_images=5, color_mode: Optional[str] = 'gray', normalize=False):
    
    iterator = iter(dataset)
    images, labels = [], []

    for _ in range(num_images // 32 + 1):
        batch_images, batch_labels = next(iterator)
        images.append(batch_images)
        labels.append(batch_labels)
        
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)

    if normalize: images = images / 255.0

    indices = np.random.choice(len(images), size=num_images, replace=False)

    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(indices):
        plt.subplot(1, num_images, i+1)
        plt.imshow(images[idx], cmap=color_mode)
        plt.title(f"Label: {np.argmax(labels[idx])}")
        plt.axis('off')
    
    plt.show()