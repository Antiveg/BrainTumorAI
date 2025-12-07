from typing import Tuple
import tensorflow as tf
from tensorflow.data import Dataset
import keras

def preprocess_data(dataset: Dataset, 
                    img_size: Tuple[int, int], 
                    normalize: bool = True,
                    model_type: str = '') -> Dataset:

    def normalize_fn(x, model_type):
        if not normalize:
            return x
        elif model_type == 'mobilenetv2':
            return keras.applications.mobilenet_v2.preprocess_input(x)
        elif model_type == 'vgg16':
            return keras.applications.vgg16.preprocess_input(x)
        elif model_type == 'resnet50v2':
            return keras.applications.resnet50.preprocess_input(x)
        else:
            return x / 255.0

    def preprocess_fn(x, y):
        x = tf.image.resize(x, img_size)
        x = normalize_fn(x, model_type)
        return x, y

    dataset = dataset.map(preprocess_fn)
    return dataset
