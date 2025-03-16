# dataset.py

import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# Define a data augmentation pipeline
# Adjust parameters as needed
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

def get_image_paths_and_labels(data_dir):
    """
    data_dir structure:
    ├─ Bleached_Hard/
    ├─ Bleached_Soft/
    ├─ Healthy_Hard/
    ├─ Healthy_Soft/
    """
    image_paths = []
    labels_hard_soft = []
    labels_bleached_healthy = []
    
    subfolders = ["Bleached_Hard", "Bleached_Soft", "Healthy_Hard", "Healthy_Soft"]
    
    for subfolder in subfolders:
        folder_path = os.path.join(data_dir, subfolder)
        for img_file in glob.glob(folder_path + "/*.jpg"):
            image_paths.append(img_file)

            # Hard vs Soft
            if "Hard" in subfolder:
                hs_label = 1  # Hard
            else:
                hs_label = 0  # Soft

            # Bleached vs Healthy
            if "Bleached" in subfolder:
                bh_label = 1  # Bleached
            else:
                bh_label = 0  # Healthy

            labels_hard_soft.append(hs_label)
            labels_bleached_healthy.append(bh_label)
    
    data_df = pd.DataFrame({
        'image_path': image_paths,
        'label_hard_soft': labels_hard_soft,
        'label_bleached_healthy': labels_bleached_healthy
    })
    return data_df

def parse_image_label(image_path, label_hard_soft, label_bleached_healthy, img_size=(224,224), augment=True):
    """
    Reads an image from 'image_path', decodes, resizes, normalizes, 
    and returns (image, (label_hard_soft, label_bleached_healthy)).
    If augment=True, apply data augmentation.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, img_size)
    image = image / 255.0  # normalize to [0,1]

    # Only apply augmentation to training samples
    if augment:
        image = data_augmentation(image)

    return image, (label_hard_soft, label_bleached_healthy)

def create_datasets(data_dir, img_size=(224,224), batch_size=32, 
                    val_split=0.15, test_split=0.15, seed=42):
    # 1. Get DataFrame of all images + labels
    data_df = get_image_paths_and_labels(data_dir)
    data_df = data_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # 2. Split into train/val/test
    train_df, test_df = train_test_split(data_df, test_size=test_split, random_state=seed)
    val_ratio = val_split / (1.0 - test_split)
    train_df, val_df = train_test_split(train_df, test_size=val_ratio, random_state=seed)

    def df_to_dataset(df, shuffle=True, is_training=False):
        image_paths = df["image_path"].values
        labels_hard_soft = df["label_hard_soft"].values
        labels_bleached_healthy = df["label_bleached_healthy"].values

        ds = tf.data.Dataset.from_tensor_slices((image_paths, labels_hard_soft, labels_bleached_healthy))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(df), seed=seed)

        ds = ds.map(
            lambda path, hs, bh: parse_image_label(path, hs, bh, img_size, augment=is_training),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = df_to_dataset(train_df, shuffle=True, is_training=True)
    val_ds   = df_to_dataset(val_df,   shuffle=False, is_training=False)
    test_ds  = df_to_dataset(test_df,  shuffle=False, is_training=False)

    return train_ds, val_ds, test_ds, train_df, val_df, test_df


# import os
# import glob
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras import layers
# from sklearn.model_selection import train_test_split

# # Define a data augmentation pipeline with additional transformations
# data_augmentation = tf.keras.Sequential([
#     layers.RandomFlip("horizontal"),
#     layers.RandomRotation(0.1),
#     layers.RandomZoom(0.1),
#     layers.RandomTranslation(height_factor=0.1, width_factor=0.1),  # Random shifting
#     layers.RandomContrast(0.1),  # Contrast adjustments
#     layers.RandomBrightness(factor=0.2),  # Brightness variation
#     layers.RandomCrop(200, 200)  # Random cropping before resizing
# ])

# # Function to normalize image based on dataset statistics
# def normalize_image(image):
#     mean = tf.constant([0.485, 0.456, 0.406])  # Example ImageNet means
#     std = tf.constant([0.229, 0.224, 0.225])  # Example ImageNet std deviations
#     return (image - mean) / std

# def get_image_paths_and_labels(data_dir):
#     image_paths = []
#     labels_hard_soft = []
#     labels_bleached_healthy = []
    
#     subfolders = ["Bleached_Hard", "Bleached_Soft", "Healthy_Hard", "Healthy_Soft"]
    
#     for subfolder in subfolders:
#         folder_path = os.path.join(data_dir, subfolder)
#         for img_file in glob.glob(folder_path + "/*.jpg"):
#             image_paths.append(img_file)

#             if "Hard" in subfolder:
#                 hs_label = 1  # Hard
#             else:
#                 hs_label = 0  # Soft

#             if "Bleached" in subfolder:
#                 bh_label = 1  # Bleached
#             else:
#                 bh_label = 0  # Healthy

#             labels_hard_soft.append(hs_label)
#             labels_bleached_healthy.append(bh_label)
    
#     data_df = pd.DataFrame({
#         'image_path': image_paths,
#         'label_hard_soft': labels_hard_soft,
#         'label_bleached_healthy': labels_bleached_healthy
#     })
#     return data_df

# def parse_image_label(image_path, label_hard_soft, label_bleached_healthy, img_size=(224,224), augment=True, use_grayscale=True):
#     """
#     Reads an image, decodes, resizes, normalizes, and applies optional grayscale conversion.
#     Ensures final image size is consistent with model input.
#     """
#     image = tf.io.read_file(image_path)
#     image = tf.image.decode_jpeg(image, channels=3)

#     if use_grayscale:
#         image = tf.image.rgb_to_grayscale(image)
#         image = tf.image.grayscale_to_rgb(image)

#     # Resize first to ensure consistent dimensions
#     image = tf.image.resize(image, img_size)

#     if augment:
#         image = data_augmentation(image)

#     # Resize again if augmentation changed size (fix random crop issue)
#     image = tf.image.resize(image, img_size)

#     # Normalize
#     image = image / 255.0  
#     image = normalize_image(image)

#     return image, (label_hard_soft, label_bleached_healthy)


# def create_datasets(data_dir, img_size=(224,224), batch_size=32, val_split=0.15, test_split=0.15, seed=42, use_grayscale=False):
#     data_df = get_image_paths_and_labels(data_dir)
#     data_df = data_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
#     train_df, test_df = train_test_split(data_df, test_size=test_split, random_state=seed)
#     val_ratio = val_split / (1.0 - test_split)
#     train_df, val_df = train_test_split(train_df, test_size=val_ratio, random_state=seed)

#     def df_to_dataset(df, shuffle=True, is_training=False):
#         image_paths = df["image_path"].values
#         labels_hard_soft = df["label_hard_soft"].values
#         labels_bleached_healthy = df["label_bleached_healthy"].values

#         ds = tf.data.Dataset.from_tensor_slices((image_paths, labels_hard_soft, labels_bleached_healthy))
#         if shuffle:
#             ds = ds.shuffle(buffer_size=len(df), seed=seed)

#         ds = ds.map(
#             lambda path, hs, bh: parse_image_label(path, hs, bh, img_size, augment=is_training, use_grayscale=use_grayscale),
#             num_parallel_calls=tf.data.AUTOTUNE
#         )
#         ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
#         return ds

#     train_ds = df_to_dataset(train_df, shuffle=True, is_training=True)
#     val_ds   = df_to_dataset(val_df,   shuffle=False, is_training=False)
#     test_ds  = df_to_dataset(test_df,  shuffle=False, is_training=False)

#     return train_ds, val_ds, test_ds, train_df, val_df, test_df
