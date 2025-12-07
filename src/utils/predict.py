import tensorflow as tf
import numpy as np
from src.data_preprocessing.basic_preprocessing import preprocess_data

def predict(model, image, class_names, model_type='vgg16', img_size=(224,224)):

    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(np.array(image), dtype=tf.float32)

    dataset = tf.data.Dataset.from_tensors((image, 0))
    preprocessed_dataset = preprocess_data(dataset, img_size=img_size, normalize=True, model_type=model_type)
    
    img_preprocessed = None
    for x, y in preprocessed_dataset:
        img_preprocessed = x

    img_batch = tf.expand_dims(img_preprocessed, axis=0)

    predictions = model.predict(img_batch)
    probs = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}
    return probs