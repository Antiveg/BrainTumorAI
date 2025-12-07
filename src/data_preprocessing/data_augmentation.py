from keras import layers, Sequential
from tensorflow.data import Dataset

def augment_data(dataset: Dataset) -> Dataset:
    
    data_augmentation = Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
    ])
    
    dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y))
    return dataset