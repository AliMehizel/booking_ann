import tensorflow as tf
from tensorflow.keras import layers, Sequential
from keras_tuner import HyperParameters

def build_model(hp, config: Config):
    """
    Builds and compiles a Keras model based on the given hyperparameters and configuration.
    
    Args:
    hp (HyperParameters): The hyperparameters to tune.
    config (Config): The configuration object containing hyperparameter ranges and other settings.
    
    Returns:
    model_tuner (Sequential): The compiled Keras model.
    """
    model_tuner = Sequential()
    model_tuner.add(layers.Dense(20, input_dim=config.input_dim, activation='relu'))
    # Number of hidden layers
    for i in range(hp.Int('num_layers', config.min_layer, config.max_layer)):
        model_tuner.add(layers.Dense(units=hp.Int('units_' + str(i),
                                                  min_value=config.min_value,
                                                  max_value=config.max_value,
                                                  step=config.step),
                                     activation=hp.Choice('activation_' + str(i), config.activation )))
    
    # Output layer for binary classification
    model_tuner.add(layers.Dense(config.out_dim, activation=config.out_activation))
    
    # Compile the model with the specified loss function and metrics
    model_tuner.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice('learning_rate', config.learning_rate)),
        loss=config.loss,
        metrics=config.metrics)
    
    return model_tuner