import tensorflow as tf
from typing import Optional

def optimize_dataset(
    dataset: tf.data.Dataset, 
    batch_size: Optional[int] = None, 
    buffer_size: int = 1000, 
    shuffle: bool = False, 
    prefetch: bool = False
) -> tf.data.Dataset:

    first_element = next(iter(dataset))
    is_batched = isinstance(first_element, tuple) and isinstance(first_element[0], tf.Tensor)
    
    if batch_size and not is_batched: dataset = dataset.batch(batch_size)
    if shuffle: dataset = dataset.shuffle(buffer_size)
    if prefetch: dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset