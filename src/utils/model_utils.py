import tensorflow_model_optimization as tfmot
import tensorflow as tf
from keras.models import load_model
import numpy as np
from src.data_preprocessing.basic_preprocessing import preprocess_data
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models

def prune_model(model_path, pruning_percentage=0.5, save_path=None):
    model = load_model(model_path)
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    pruning_schedule = tfmot.sparsity.keras.ConstantSparsity(pruning_percentage, 0)

    head_classifier_layers = [layer for layer in model.layers if isinstance(layer, (layers.Dense, layers.Dropout))]
    new_model_layers = []
    for layer in model.layers:
        if layer in head_classifier_layers:
            print(f"Pruning layer: {layer.name}")
            new_model_layers.append(prune_low_magnitude(layer, pruning_schedule=pruning_schedule))
        else:
            new_model_layers.append(layer)

    pruned_model = tf.keras.models.Model(inputs=model.input, outputs=new_model_layers[-1].output)
    pruned_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    if save_path:
        pruned_model.save(save_path)
        print(f"Pruned model saved to {save_path}")

    return pruned_model

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