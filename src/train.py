# train.py
import tensorflow as tf
from dataset import create_datasets
from model import build_optimized_dual_model

def main():
    # --- Configuration ---
    data_dir = "data"
    img_size = (224, 224)
    batch_size = 32
    
    # How many epochs in each phase
    epochs_stage1 = 12   # You can pick 5, 10, or more
    epochs_stage2 = 16  # Additional fine-tuning epochs

    # --- 1. Create Datasets ---
    train_ds, val_ds, test_ds, _, _, _ = create_datasets(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size,
        val_split=0.15,
        test_split=0.15
    )
    
    # --- 2. Build Model (MobileNetV2 is Frozen by Default) ---
    model = build_optimized_dual_model(input_shape=(img_size[0], img_size[1], 3))
    # In build_optimized_dual_model(), your code sets base_model.trainable = False
    # So the entire MobileNet is currently frozen.

    # --- 3. Compile: Stage 1 with a moderate LR ---
    initial_lr = 1e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    model.compile(
        optimizer=optimizer,
        loss={
            'hard_soft_output': 'binary_crossentropy',
            'bleached_healthy_output': 'binary_crossentropy'
        },
        metrics={
            'hard_soft_output': ['accuracy'],
            'bleached_healthy_output': ['accuracy']
        }
    )
    
    # Callback(s)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # --- 4. Stage 1 Training ---
    print("\n=== Stage 1: Training with Frozen Base ===")
    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_stage1,
        callbacks=[early_stop]
    )
    
    # Evaluate quickly after Stage 1
    print("\nStage 1 Evaluation on Test Set:")
    results_stage1 = model.evaluate(test_ds, return_dict=True)
    print(results_stage1)
    
    # --- 5. Stage 2: Fine-Tuning ---
    print("\n=== Stage 2: Fine-Tuning the Top Layers ===")

    # Get all layers
    all_layers = model.layers
    
    # Find where base model ends (before our custom layers)
    base_layers = all_layers[:-5]  # Exclude final GAP, dropouts, and dense layers
    
    # Make model trainable
    model.trainable = True
    
    # Freeze all base layers except last 20
    for layer in base_layers[:-20]:
        layer.trainable = False
    for layer in base_layers[-20:]:
        layer.trainable = True

    # Re-compile with smaller learning rate
    fine_tune_lr = 1e-5
    optimizer_finetune = tf.keras.optimizers.Adam(learning_rate=fine_tune_lr)
    model.compile(
        optimizer=optimizer_finetune,
        loss={
            'hard_soft_output': 'binary_crossentropy',
            'bleached_healthy_output': 'binary_crossentropy'
        },
        metrics={
            'hard_soft_output': ['accuracy'],
            'bleached_healthy_output': ['accuracy']
        }
    )

    # Fit again for Stage 2
    history_stage2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_stage2,
        callbacks=[early_stop]
    )
    
    # Evaluate after Fine-Tuning
    print("\nStage 2 Evaluation on Test Set:")
    results_stage2 = model.evaluate(test_ds, return_dict=True)
    print(results_stage2)
    
    # --- 6. Save Fine-Tuned Model ---
    model.save("dual_output_cnn_transfer_finetuned.keras")
    print("Fine-tuned model saved to dual_output_cnn_transfer_finetuned.keras")

if __name__ == "__main__":
    main()