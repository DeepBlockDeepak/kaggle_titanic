import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential


def build_model(input_shape):
    """
    Constructs and compiles a Sequential model for binary classification.

    Parameters:
    - input_shape (int): The number of features in the input dataset.

    Returns:
    - Compiled TensorFlow Keras Sequential model.
    """
    model = Sequential(
        [
            # Dense layer with ReLU activation, initializing the model with the given input shape
            Dense(64, activation="relu", input_shape=(input_shape,)),
            # Dropout layer to reduce overfitting by randomly setting input units to 0 during training
            Dropout(0.5),
            # Output layer with a single neuron and sigmoid activation function for binary classification
            Dense(1, activation="sigmoid"),
        ]
    )
    # Compile the model with Adam optimizer and binary crossentropy loss, tracking accuracy metric
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def tf_keras_main(X_train, y_train, X_val, y_val, return_scaler=True):
    """
    Trains a TensorFlow Keras model on the provided training data and evaluates it on validation data.

    Parameters:
    - X_train: Training feature data.
    - y_train: Training target data (labels).
    - X_val: Validation feature data.
    - y_val: Validation target data (labels).
    - return_scaler (bool, optional): Whether to return the StandardScaler instance used for scaling.

    Returns:
    - model: Trained TensorFlow Keras model.
    - predictions: Predictions made by the model on the validation dataset.
    - scaler (optional): The StandardScaler instance used for feature normalization.
    """
    # Normalize the features using StandardScaler to ensure features are on a similar scale
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Build the model based on the scaled training data's shape
    model = build_model(X_train_scaled.shape[1])

    # Train the model, using early stopping to halt training when validation loss ceases to decrease
    model.fit(
        X_train_scaled,
        y_train,
        epochs=100,
        batch_size=16,
        validation_data=(X_val_scaled, y_val),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            )
        ],
    )

    # Generate probability predictions for the validation set
    predictions_proba = model.predict(X_val_scaled)
    # Convert probabilities to binary class labels based on a 0.5 threshold
    predictions = (predictions_proba > 0.5).astype("int32").flatten()

    # Save the trained model for later use or deployment
    model.save("models/titanic_keras_model")

    # Return the model, predictions, and optionally the scaler based on the function parameter
    if return_scaler:
        return model, predictions, scaler
    else:
        return model, predictions
