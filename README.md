# Coral Classification Project

This repository contains code and documentation for a dual-output coral classification project. The goal is to classify images of corals into:
1. **Hard vs. Soft**  
2. **Bleached vs. Healthy**  

This approach allows us to perform two distinct binary classifications with a single, unified model. The project leverages **transfer learning** (with MobileNetV2) and **data augmentation** to improve performance and robustness. Below you will find a comprehensive overview of how the project is structured, how to use the code, and which methods are employed.

---

## Table of Contents
1. [Background and Motivation](#background-and-motivation)
2. [Project Structure](#project-structure)
3. [Dependencies and Installation](#dependencies-and-installation)
4. [Dataset and Directory Setup](#dataset-and-directory-setup)
5. [Data Preprocessing & Augmentation](#data-preprocessing--augmentation)
6. [Model Architecture](#model-architecture)
7. [Training the Model](#training-the-model)
8. [Evaluation](#evaluation)
9. [How to Run](#how-to-run)
10. [References](#references)

---

## Background and Motivation

Coral reefs are crucial to marine biodiversity. However, climate change, pollution, and other stressors can lead to **coral bleaching** and ecosystem collapse. Identifying and classifying corals quickly (including whether they are bleached) can aid conservation efforts and provide researchers with timely data.

Drawing on methods from prior works in coral identification and using modern **convolutional neural networks (CNNs)**, this project classifies:
- **Hard vs. Soft corals**, and
- **Bleached vs. Non-bleached (Healthy) corals**.

Each image goes through a **dual-output** classification system, meaning the model simultaneously produces two separate binary labels. This project was carried out for a computer vision course project (see [References](#references) for details).

---

## Project Structure

Below is an overview of the key files in this repository:

```
.
├── dataset.py           # Dataset handling: loading, splitting, and augmenting images
├── evaluate.py          # Model evaluation: confusion matrices & classification reports
├── model.py             # Building the transfer-learning dual-output model
├── train.py             # Training script: from data loading to saving the final model
└── README.md            # You are here!
```

**Key highlights of each script**:

- **`dataset.py`**  
  - Provides two main functions:  
    1. `create_datasets()`: Splits data into training, validation, and test sets.  
    2. `parse_image_label()`: Reads images, applies augmentations, and normalizes.  
  - Demonstrates data augmentation (flip, rotation, zoom, brightness, etc.).  
  - Has the option to convert images to grayscale and back to RGB if desired.  

- **`model.py`**  
  - Contains `build_optimized_dual_model()`: a MobileNetV2-based network with two output heads (for the two binary classifications).  

- **`train.py`**  
  - Loads data using `create_datasets()`.  
  - Builds the model via `build_optimized_dual_model()`.  
  - Compiles, trains, and saves the final model as `dual_output_cnn_transfer.keras`.  

- **`evaluate.py`**  
  - Loads the saved model and the test set.  
  - Generates predictions for each of the two binary classifications.  
  - Prints confusion matrices and detailed classification reports (precision, recall, F1-score).  

---

## Dependencies and Installation

This project uses **Python 3.7+**. Install the following major packages:

- **TensorFlow 2.x**  
- **scikit-learn** (for `train_test_split` and evaluation metrics)  
- **NumPy**  
- **Pandas**  

You can install them via pip:

```bash
pip install tensorflow scikit-learn numpy pandas
```

*(Optional) If you plan to run the project on GPU, ensure you have the appropriate GPU drivers and CUDA/CuDNN libraries installed.*

---

## Dataset and Directory Setup

The scripts assume you have a folder called `data/` with **four** subfolders:

```
data
├── Bleached_Hard/
├── Bleached_Soft/
├── Healthy_Hard/
└── Healthy_Soft/
```

Each subfolder should contain `.jpg` images matching its label. For instance:
- `Bleached_Hard/` has pictures of bleached hard corals.
- `Healthy_Soft/` has pictures of healthy soft corals.

The code (`dataset.py`) will automatically label the images by:
- **Hard vs. Soft** (1 for Hard, 0 for Soft)
- **Bleached vs. Healthy** (1 for Bleached, 0 for Healthy)

When you run `create_datasets()`, the dataset is split into:
- **Train set** (~70% by default)
- **Validation set** (~15% by default)
- **Test set** (~15% by default)

You can customize the exact sizes by modifying `val_split` and `test_split` in `create_datasets()`.

---

## Data Preprocessing & Augmentation

1. **Resizing & Normalization**  
   All images are resized to `(224, 224)` and normalized to `[0, 1]`.

2. **Augmentation (Train Only)**  
   By default, the `train_ds` pipeline includes:
   - Random horizontal flips  
   - Random rotations (±10%)  
   - Random zoom (±10%)  

   The *advanced* augmentation pipeline (commented-out portion in `dataset.py`) also includes:
   - Random brightness changes  
   - Random cropping  
   - Random translation (shifts)  
   - Grayscale conversion (optional)  

3. **Dual Labels**  
   Each image has two separate binary labels:
   - `label_hard_soft`
   - `label_bleached_healthy`  

These augmentations help combat overfitting and improve model generalization.

---

## Model Architecture

We use **MobileNetV2** as our **base**, which is:
- Pre-trained on ImageNet
- Used without the top classification layers
- Frozen (non-trainable) in initial stages

On top of MobileNetV2’s output, we add:
1. **GlobalAveragePooling2D** layer  
2. **Dropout** (0.3) + Dense(128, ReLU) + Dropout(0.3)  
3. **Two Output Heads** (each a single-unit Dense layer with sigmoid activation):  
   - `hard_soft_output`  
   - `bleached_healthy_output`  

This design is found in `model.py`, within the function `build_optimized_dual_model()`.

---

## Training the Model

1. **Compile**  
   ```python
   model.compile(
       optimizer=tf.keras.optimizers.Adam(learning_rate=...),
       loss={
           'hard_soft_output': 'binary_crossentropy',
           'bleached_healthy_output': 'binary_crossentropy'
       },
       metrics={
           'hard_soft_output': ['accuracy'],
           'bleached_healthy_output': ['accuracy']
       }
   )
   ```
   - We use two binary crossentropy losses, one per output head.

2. **Callbacks**  
   - **Early Stopping**: Halts training if the validation loss does not improve for a given patience (e.g., 5 epochs).

3. **Learning Rate Schedule**  
   - `ExponentialDecay`:  Start with a small learning rate (e.g., `1e-4`), then reduce every certain number of steps.

4. **Run**  
   - By default, we train for up to 30 epochs or until early stopping triggers.

After training, the script **saves** the model to `dual_output_cnn_transfer.keras`.

---

## Evaluation

In `evaluate.py`, we do the following:

1. **Load** the saved model from `dual_output_cnn_transfer.keras`.
2. **Create** a test dataset with the same parameters used at training time.
3. **Generate Predictions** for each output head:
   - Hard vs. Soft
   - Bleached vs. Healthy
4. **Compute** the confusion matrices and classification reports for each task using `scikit-learn`:
   ```python
   confusion_matrix(y_true, y_pred)
   classification_report(y_true, y_pred)
   ```

This step produces metrics like **accuracy**, **precision**, **recall**, and **F1-score**.

---

## How to Run

1. **Clone** this repository:
   ```bash
   git clone https://YourUser/@github.com/GerwinMateo/CS131-FinalProject.git
   ```

2. **Install** dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare** your dataset in the `data/` folder structure:
   ```
   data
   ├── Bleached_Hard/
   ├── Bleached_Soft/
   ├── Healthy_Hard/
   └── Healthy_Soft/
   ```

4. **Train** the model:
   ```bash
   python src/train.py
   ```
   - This will train the dual-output model and save it to `dual_output_cnn_transfer.keras`.

5. **Evaluate** the model on the test set:
   ```bash
   python src/evaluate.py
   ```
   - This loads `dual_output_cnn_transfer.keras` and prints confusion matrices / classification reports for both tasks.

---

## References

- **Project Specification**:  
  See **CS131 Project.pdf** for the original project description and background.  
    https://docs.google.com/document/d/1ULIov0SK_3703Mkj-eQP-F6OJzKyent0_9p6ZNMXcdA/edit?usp=sharing

- **MobileNetV2**:  
  [Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks* (CVPR, 2018)](https://arxiv.org/abs/1801.04381)

- **TensorFlow/Keras Documentation**:  
  [https://www.tensorflow.org/api_docs/python/tf/keras](https://www.tensorflow.org/api_docs/python/tf/keras)

---