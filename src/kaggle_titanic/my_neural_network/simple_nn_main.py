from my_neural_network import SimpleNeuralNetwork

def simple_nn_main(X_train, y_train, X_val, y_val):
    # Transform data into NumPy arrays if not already done in preprocess
    X_train_np = X_train.to_numpy()
    X_val_np = X_val.to_numpy()
    y_train_np = y_train.to_numpy().reshape(1, -1)  # Reshape for compatibility with your NN implementation
    y_val_np = y_val.to_numpy().reshape(1, -1)

    # Initialize the neural network
    nn = SimpleNeuralNetwork(layer_dims=[X_train_np.shape[1], 10, 1])
    
    # Train the neural network
    nn.train(X_train_np, y_train_np, iterations=1000, learning_rate=0.01)

    # Predict on validation set
    AL_val, _ = nn.forward_propagation(X_val_np)
    predictions = (AL_val > 0.5).astype(int)  # Convert probabilities to 0 or 1

    return nn, predictions.flatten()
