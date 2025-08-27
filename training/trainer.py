import numpy as np
import tensorflow as tf

class SimpleTrainer:
    """A minimalistic trainer to simulate weight updates."""

    def __init__(self, input_dim: int = 10):
        self.weights = np.random.randn(input_dim).tolist()

    def train_step(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Simulate a single training step and update weights.
        (Replace with real ML logic later)
        """
        # Simple simulation: gradient = mean error
        preds = X.dot(np.array(self.weights))
        error = preds - y
        grad = X.T.dot(error) / len(y)

        # Simple SGD update
        lr = 0.01
        self.weights = (np.array(self.weights) - lr * grad).tolist()

        return {
            "loss": float(np.mean(error**2)),
            "weights": self.weights
        }

def train_model(params, train_data, val_data, job_id, ps_client):
    epochs = int(params['epochs'])
    batch_size = int(params['batch_size'])
    learning_rate = float(params['learning_rate'])
    optimizer_name = params['optimizer']
    loss_function = params['loss_function']
    validation_split = float(params['validation_split'])
    early_stopping_patience = int(params['early_stopping_patience'])
    dropout_rate = float(params['dropout_rate'])
    random_seed = int(params['random_seed'])

    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    base_model = tf.keras.applications.DenseNet121(
        input_shape=(512, 512, 3),
        include_top=False,
        weights=None
    )
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    output = tf.keras.layers.Dense(train_data.num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=output)

    optimizer = getattr(tf.keras.optimizers, optimizer_name.capitalize())(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        patience=early_stopping_patience,
        restore_best_weights=True
    )

    training_log_history = []
    for epoch in range(epochs):
        history = model.fit(
            train_data,
            epochs=1,
            batch_size=batch_size,
            validation_data=val_data,
            callbacks=[early_stopping],
            verbose=0
        )
        logs = {
            "loss": float(history.history["loss"][-1]),
            "accuracy": float(history.history["accuracy"][-1]),
            "precision": float(history.history["precision"][-1]),
            "recall": float(history.history["recall"][-1]),
            "auc": float(history.history["auc"][-1]),
            "val_loss": float(history.history["val_loss"][-1]),
            "val_accuracy": float(history.history["val_accuracy"][-1]),
            "val_precision": float(history.history["val_precision"][-1]),
            "val_recall": float(history.history["val_recall"][-1]),
            "val_auc": float(history.history["val_auc"][-1])
        }
        log_entry = {"epoch": epoch+1, "logs": logs}
        training_log_history.append(log_entry)
        ps_client.send_log(job_id, log_entry, training_log_history)
