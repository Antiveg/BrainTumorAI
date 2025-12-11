from keras.applications import ResNet50, VGG16, MobileNetV2
from keras import layers, models
from keras.models import Model
from typing import Optional, Callable, Tuple

def create_head_classifier(input_tensor, num_classes: int) -> models.Sequential:
    head_classifier = models.Sequential([
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return head_classifier(input_tensor)

def create_resnet50_model(
        input_shape: Tuple[int, int, int] = (224, 224, 3), 
        fine_tune: bool = False,
        head_classifier: Optional[Callable] = None,
        num_classes: int = 2,
        show_info: bool = False) -> Model:

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    base_model.trainable = False
    if fine_tune:
        for layer in base_model.layers[-4:]:
            layer.trainable = True
    
    x = base_model.output
    if head_classifier:
        head_model = head_classifier(x, num_classes)
    else:
        x = layers.GlobalAveragePooling2D()(x)
        head_model = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=base_model.input, outputs=head_model)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    if show_info: model.summary()
    return model

def create_vgg16_model(input_shape: Tuple[int, int, int] = (224, 224, 3), 
                       fine_tune: bool = False,
                       head_classifier: Optional[Callable] = None,
                       num_classes: int = 2,
                       show_info: bool = False) -> Model:

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    
    base_model.trainable = False
    if fine_tune:
        for layer in base_model.layers[-4:]:
            layer.trainable = True
    
    x = base_model.output
    if head_classifier:
        head_model = head_classifier(x, num_classes)
    else:
        x = layers.GlobalAveragePooling2D()(x)
        head_model = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=base_model.input, outputs=head_model)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    if show_info: model.summary()
    return model

def create_mobilenetv2_model(input_shape: Tuple[int, int, int] = (224, 224, 3), 
                             fine_tune: bool = False,
                             head_classifier: Optional[Callable] = None,
                             num_classes: int = 2) -> Model:

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    
    base_model.trainable = False
    if fine_tune:
        for layer in base_model.layers[-4:]:
            layer.trainable = True
    
    x = base_model.output
    if head_classifier:
        head_model = head_classifier(x, num_classes)
    else:
        x = layers.GlobalAveragePooling2D()(x)
        head_model = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=base_model.input, outputs=head_model)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model