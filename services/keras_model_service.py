from tensorflow.keras.applications import (
    DenseNet121, DenseNet169, DenseNet201,
    EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3,
    EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7,
    InceptionV3, MobileNet, MobileNetV2, NASNetLarge, NASNetMobile,
    ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, ResNet152V2,
    VGG16, VGG19, Xception
)
import tensorflow as tf
from tensorflow.keras import layers, models, Input

class KerasCatalogService:
    MODELS_DICT = {
        "DenseNet121": DenseNet121,
        "DenseNet169": DenseNet169,
        "DenseNet201": DenseNet201,
        "EfficientNetB0": EfficientNetB0,
        "EfficientNetB1": EfficientNetB1,
        "EfficientNetB2": EfficientNetB2,
        "EfficientNetB3": EfficientNetB3,
        "EfficientNetB4": EfficientNetB4,
        "EfficientNetB5": EfficientNetB5,
        "EfficientNetB6": EfficientNetB6,
        "EfficientNetB7": EfficientNetB7,
        "InceptionV3": InceptionV3,
        "MobileNet": MobileNet,
        "MobileNetV2": MobileNetV2,
        "NASNetLarge": NASNetLarge,
        "NASNetMobile": NASNetMobile,
        "ResNet50": ResNet50,
        "ResNet50V2": ResNet50V2,
        "ResNet101": ResNet101,
        "ResNet101V2": ResNet101V2,
        "ResNet152": ResNet152,
        "ResNet152V2": ResNet152V2,
        "VGG16": VGG16,
        "VGG19": VGG19,
        "Xception": Xception
    }

    @staticmethod
    def get_model_object__(algo_name: str, input_shape=(150, 150, 3)):
        if algo_name not in KerasCatalogService.MODELS_DICT:
            raise ValueError(f"Algorithm '{algo_name}' is not available. Choose from {list(KerasCatalogService.MODELS_DICT)}")

        model_constructor = KerasCatalogService.MODELS_DICT[algo_name]
        
        # Return the model exactly as defined in keras.applications
        return model_constructor(weights='imagenet', include_top=True, input_shape=input_shape)
    

    @staticmethod
    def get_model_object(algo_name: str, input_shape=(150, 150, 3), include_top=False):
        if algo_name not in KerasCatalogService.MODELS_DICT:
            raise ValueError(f"Algorithm '{algo_name}' is not available.")
        model_constructor = KerasCatalogService.MODELS_DICT[algo_name]
        return model_constructor(weights='imagenet', include_top=include_top, input_shape=input_shape)
    


    @staticmethod
    def build_model(algo: str, input_shape):
        if not input_shape:
            input_shape = (150, 150, 3)

        try:
            base_model = KerasCatalogService.get_model_object(algo, input_shape=input_shape, include_top=False)
            base_model.trainable = False

            inputs = tf.keras.Input(shape=input_shape)
            x = base_model(inputs, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
            return tf.keras.Model(inputs, outputs)

        except ValueError:
            # Fallback model
            inputs = tf.keras.Input(shape=input_shape)
            x = tf.keras.layers.Conv2D(16, 3, activation="relu")(inputs)
            x = tf.keras.layers.MaxPool2D()(x)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
            return tf.keras.Model(inputs, outputs)