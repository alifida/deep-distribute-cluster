training_service.py# training/training_service.py
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json, random, numpy as np
from db.utils import SessionLocal
from db.models import TrainTrainingJob

class DBLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, job_id):
        super().__init__()
        self.job_id = job_id

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        session = SessionLocal()
        try:
            job = session.query(TrainTrainingJob).filter_by(id=self.job_id).first()
            if job:
                history = json.loads(job.training_log_history) if job.training_log_history else []
                epoch_log = {"epoch": epoch+1, "logs": {k: float(v) for k, v in logs.items()}}
                history.append(epoch_log)
                job.training_log = json.dumps(epoch_log)
                job.training_log_history = json.dumps(history)
                session.add(job)
                session.commit()
        finally:
            session.close()

def build_model(num_classes, dropout_rate, optimizer_name, learning_rate, loss_function, random_seed):
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    base = DenseNet121(weights="imagenet", include_top=False, input_shape=(512, 512, 3))
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(dropout_rate)(x)
    output = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=base.input, outputs=output)

    opt = optimizers.Adam(learning_rate=learning_rate) if optimizer_name.lower() == "adam" else optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss=loss_function, metrics=["accuracy", tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model

def get_data(train_path, validation_split, batch_size):
    gen = ImageDataGenerator(rescale=1./255, validation_split=validation_split)
    train_gen = gen.flow_from_directory(train_path, target_size=(512, 512), batch_size=batch_size, subset="training")
    val_gen = gen.flow_from_directory(train_path, target_size=(512, 512), batch_size=batch_size, subset="validation")
    return train_gen, val_gen

"""
def train_model(job_data, dataset_details):
    job_id = job_data["job_id"]
    p = job_data["parameter_settings"]
    model = build_model(dataset_details["train"]["total_classes"], float(p["dropout_rate"]), p["optimizer"], float(p["learning_rate"]), p["loss_function"], int(p["random_seed"]))
    train_gen, val_gen = get_data(dataset_details["train"]["root_path"], float(p["validation_split"]), int(p["batch_size"]))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=int(p["early_stopping_patience"]), restore_best_weights=True),
        DBLoggingCallback(job_id)
    ]

    model.fit(train_gen, validation_data=val_gen, epochs=int(p["epochs"]), callbacks=callbacks)
    return model

"""