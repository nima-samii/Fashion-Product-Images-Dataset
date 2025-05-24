import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.applications import ResNet50 # type: ignore
from tensorflow.keras import layers, Model # type: ignore

def efficientnet_model(input_shape=None, num_classes=None, freeze_base=True, dropout_rate=0.5):
    # Loading the base model (EfficientNet-B4)
    base_model = keras.applications.EfficientNetB4(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freezing or not freezing base layers
    base_model.trainable = not freeze_base
    
    # Create a new head for classification
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)  # training=False for BatchNorm in the first phase
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Making the final model
    model = Model(inputs, outputs)
    
    return model


def resnet_model(input_shape=None, num_classes=None, freeze_base=True, dropout_rate=0.3, pr_dir=''):
    weight_dir = os.path.join(
            pr_dir, 'Models', 'Weights', 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

    # Load ResNet50 base model
    base_model = ResNet50(
        weights=weight_dir,
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = not freeze_base

    # Create the new head
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)  # Important if freeze_base=True
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    # x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Build the model
    model = Model(inputs, outputs)
    return model
