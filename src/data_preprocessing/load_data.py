from typing import Optional, Tuple
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing import image
import numpy as np
import os

def load_data(train_dir: str, 
              test_dir: str,
              img_size: Tuple[int, int] = (224, 224), 
              batch_size: int = 32,
              validation_dir: Optional[str] = None,
              validation_split: Optional[float] = 0.2,
              seed: int = 42,
              color_mode: Optional[str] = 'grayscale') -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    
    trainset = image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical',
        validation_split=validation_split,
        subset='training',
        shuffle=True,
        seed=seed,
        color_mode=color_mode,
    )

    if validation_dir is not None:
        validset = image_dataset_from_directory(
            validation_dir,
            image_size=img_size,
            batch_size=batch_size,
            label_mode='categorical',
            shuffle=False,
            seed=seed,
            color_mode=color_mode,
        )
    else:
        validset = image_dataset_from_directory(
            train_dir,
            image_size=img_size,
            batch_size=batch_size,
            label_mode='categorical',
            validation_split=validation_split,
            subset='validation',
            shuffle=True,
            seed=seed,
            color_mode=color_mode,
        )

    testset = image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=False,
        seed=seed,
        color_mode=color_mode,
    )

    return trainset, validset, testset