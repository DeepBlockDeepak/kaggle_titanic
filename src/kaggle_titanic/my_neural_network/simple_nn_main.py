from my_neural_network import SimpleNeuralNetwork
from sklearn.preprocessing import StandardScaler


def simple_nn_main(X_train, y_train, X_val, y_val):
    # normalize numerical features
    numerical_features = X_train.select_dtypes(
        include=["float64", "int64"]
    ).columns.tolist()
    scaler = StandardScaler()
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_val[numerical_features] = scaler.transform(X_val[numerical_features])

    # convert boolean features to integer type
    boolean_features = X_train.select_dtypes(include=["bool"]).columns
    X_train[boolean_features] = X_train[boolean_features].astype(int)
    X_val[boolean_features] = X_val[boolean_features].astype(int)

    # transform data into NumPy arrays and ensure they are transposed correctly
    X_train_np = X_train.to_numpy().T  # transpose: features as rows, samples as columns
    X_val_np = X_val.to_numpy().T  # transpose: same as X_train_np
    y_train_np = y_train.to_numpy().reshape(1, -1)  # reshape to (1, number of samples)

    # init the network
    nn = SimpleNeuralNetwork(layer_dims=[X_train_np.shape[0], 10, 1])

    # train
    nn.train(X_train_np, y_train_np, iterations=1000, learning_rate=0.01)

    # predict on validation set
    AL_val, _ = nn.forward_propagation(X_val_np)
    predictions = (AL_val > 0.5).astype(int)  # convert probabilities to 0 or 1

    return nn, predictions.flatten()
