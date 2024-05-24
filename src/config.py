


class Config:
    """
    A class to define hyperparameter ranges and other configurations for the model.
    
    Attributes:
    min_layer (int): Minimum number of hidden layers.
    max_layer (int): Maximum number of hidden layers.
    min_value (int): Minimum number of units in a hidden layer.
    max_value (int): Maximum number of units in a hidden layer.
    step (int): Step size for the number of units.
    learning_rate (list): List of learning rates to choose from.
    loss (str): Loss function to use.
    metrics (list): List of metrics to evaluate the model.
    activation (list): List of activation functions to choose from for hidden layers.
    out_dim (int): Number of output units.
    out_activation (str): Activation function for the output layer.
    """
    def __init__(self, input_dim,min_layer, max_layer, min_value, max_value, 
                 learning_rate=[1e-2, 1e-3, 1e-4],
                 activation=['relu', 'tanh'],
                 loss='binary_crossentropy', metrics=['accuracy'],
                 out_activation='sigmoid', step=32, out_dim=1) -> None:
        self.input_dim = input_dim
        self.min_layer = min_layer
        self.max_layer = max_layer
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.learning_rate = learning_rate
        self.loss = loss
        self.metrics = metrics
        self.activation = activation
        self.out_dim = out_dim
        self.out_activation = out_activation
