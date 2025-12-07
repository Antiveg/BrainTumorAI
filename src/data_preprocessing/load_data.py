from typing import Optional, Tuple
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory

def load_data(dataset_dir: str, 
              img_size: Tuple[int, int] = (224, 224), 
              batch_size: int = 32,
              training_dir: Optional[str] = None, 
              validation_dir: Optional[str] = None, 
              testing_dir: Optional[str] = None,
              validation_split: Optional[float] = 0.2,
              seed: int = 42,
              color_mode: Optional[str] = 'grayscale') -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    
    if training_dir is None: training_dir = dataset_dir
    
    trainset = image_dataset_from_directory(
        training_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical',
        validation_split=validation_split,
        subset='training',
        shuffle=True,
        seed=seed,
        color_mode=color_mode,
    )

    if validation_dir is None:
        validset = image_dataset_from_directory(
            dataset_dir,
            image_size=img_size,
            batch_size=batch_size,
            label_mode='categorical',
            validation_split=validation_split,
            subset='validation',
            shuffle=False,
            seed=seed,
            color_mode=color_mode,
        )
    else:
        validset = image_dataset_from_directory(
            validation_dir,
            image_size=img_size,
            batch_size=batch_size,
            label_mode='categorical',
            shuffle=False,
            seed=seed,
            color_mode=color_mode,
        )

    if testing_dir is not None:
        testset = image_dataset_from_directory(
            testing_dir,
            image_size=img_size,
            batch_size=batch_size,
            label_mode='categorical',
            shuffle=False,
            seed=seed,
            color_mode=color_mode,
        )
    else:
        testset = []

    return trainset, validset, testset