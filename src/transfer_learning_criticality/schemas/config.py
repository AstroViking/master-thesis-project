from schema import Schema, And, Or, Use, Optional, SchemaError

config_schema = Schema({
    "experiments": [{
        "dataset": Or("MNIST", "FashionMNIST", "CIFAR10"),
        "model": Or("FeedForwardNet", "ConvolutionalNet"),
        "model_parameters": {
            "hidden_layer_width": Use(int),
            "num_hidden_layers": Use(int),
            "non_linearity": Or("tanh", "relu", "selu")
        },
        "num_epochs": Use(int),
        "batch_size": Use(int),
        "learning_rate": Use(float),
        "num_samples_per_class": Use(int)
    }]
})