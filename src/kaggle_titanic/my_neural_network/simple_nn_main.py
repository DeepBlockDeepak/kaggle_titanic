from my_neural_network import NeuralNetworkConfig, SimpleNeuralNetwork

from kaggle_titanic.evaluate_model import evaluate_model


def simple_nn_main(X_train, y_train, X_val, y_val):

    # init the neural network
    nnConfig = NeuralNetworkConfig(
        layer_dims=[X_train.shape[0], 40, 1], optimizer="adam"
    )
    nn = SimpleNeuralNetwork(nnConfig)

    # train
    nn.train(X_train, y_train, epochs=1000)

    # predict on validation set
    predictions = evaluate_model(nn, X_val, y_val)
    predictions = (predictions > 0.5).astype(int)  # convert probabilities to 0 or 1

    return nn, predictions.flatten()
