# model.py

import tensorflow as tf
from tensorflow.keras import layers, models

def build_optimized_dual_model(input_shape=(224,224,3)):
    """
    Transfer Learning from MobileNetV2:
      - Pretrained on ImageNet
      - Freeze the base initially
      - Add custom layers for dual binary outputs
    """
    # Load MobileNetV2 as the backbone model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    # Freeze base model for faster, stable initial training
    base_model.trainable = True

    x = base_model.output
    # Global avg pooling to reduce dimensions
    x = layers.GlobalAveragePooling2D()(x)
    
    # Custom classification head
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # Two binary outputs
    hard_soft_output = layers.Dense(
        1, activation='sigmoid', name='hard_soft_output'
    )(x)
    bleached_healthy_output = layers.Dense(
        1, activation='sigmoid', name='bleached_healthy_output'
    )(x)

    model = models.Model(
        inputs=base_model.input, 
        outputs=[hard_soft_output, bleached_healthy_output],
        name="dual_output_cnn_transfer"
    )

    return model
