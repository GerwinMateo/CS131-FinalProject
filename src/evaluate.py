# evaulate.py

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(model, test_ds):
    """
    Generates predictions on the test dataset for each output head
    and prints confusion matrices + classification reports.
    """
    y_true_hard_soft = []
    y_true_bleached_healthy = []
    y_pred_hard_soft = []
    y_pred_bleached_healthy = []
    
    # Loop over the test dataset
    for images, (labels_hs, labels_bh) in test_ds:
        preds = model.predict(images)
        
        # preds is a list [preds_for_hard_soft, preds_for_bleached_healthy]
        preds_hs = preds[0].flatten()
        preds_bh = preds[1].flatten()
        
        # Convert labels/preds to numpy
        y_true_hard_soft.extend(labels_hs.numpy())
        y_true_bleached_healthy.extend(labels_bh.numpy())
        
        y_pred_hard_soft.extend((preds_hs > 0.5).astype(int))
        y_pred_bleached_healthy.extend((preds_bh > 0.5).astype(int))
    
    y_true_hs = np.array(y_true_hard_soft)
    y_true_bh = np.array(y_true_bleached_healthy)
    y_pred_hs = np.array(y_pred_hard_soft)
    y_pred_bh = np.array(y_pred_bleached_healthy)
    
    # Confusion matrix & classification report for Hard vs Soft
    print("=== Hard vs. Soft ===")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true_hs, y_pred_hs))
    print("Classification Report:")
    print(classification_report(y_true_hs, y_pred_hs, digits=4))
    
    # Confusion matrix & classification report for Bleached vs Healthy
    print("\n=== Bleached vs. Healthy ===")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true_bh, y_pred_bh))
    print("Classification Report:")
    print(classification_report(y_true_bh, y_pred_bh, digits=4))

def main():
    # 1. Load the trained model in Keras format:
    model = tf.keras.models.load_model("dual_output_cnn_transfer.keras", compile=False)
    
    # 2. Recreate the test dataset (make sure parameters match training!)
    from dataset import create_datasets
    data_dir = "data"
    img_size = (224, 224)
    batch_size = 32

    _, _, test_ds, _, _, _ = create_datasets(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size,
        val_split=0.15,
        test_split=0.15
    )
    
    # 3. Evaluate
    evaluate_model(model, test_ds)

if __name__ == "__main__":
    main()
