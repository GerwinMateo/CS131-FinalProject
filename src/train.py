# train.py

import tensorflow as tf
from dataset import create_datasets
from model import build_optimized_dual_model

def main():
    # Configuration
    data_dir = "data"
    img_size = (224, 224)
    batch_size = 32
    epochs = 30  # More epochs for better results

    # Load data
    train_ds, val_ds, test_ds, train_df, val_df, test_df = create_datasets(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size,
        val_split=0.15,
        test_split=0.15
    )
    
    # Build the transfer-learning model
    model = build_optimized_dual_model(input_shape=(img_size[0], img_size[1], 3))

    # Use a lower initial learning rate + exponential decay
    initial_lr = 1e-4
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=100,       # adjust based on dataset size
        decay_rate=0.9,        # e.g., 10% decay every 100 steps
        staircase=True
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

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

    model.summary()

    # Early stopping callback
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,            # stop if val_loss doesn't improve after 5 epochs
        restore_best_weights=True
    )

    # Fit the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stop]
    )

    # Evaluate on test set
    print("Evaluating on test set...")
    results = model.evaluate(test_ds, return_dict=True)
    print("Test Results:", results)

    # Save the model (Keras format recommended)
    model.save("dual_output_cnn_transfer.keras")
    print("Model saved to dual_output_cnn_transfer.keras")

if __name__ == "__main__":
    main()
