import torch
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimization algorithms
from sklearn.preprocessing import StandardScaler  # For feature scaling
from torch.utils.data import DataLoader, TensorDataset  # For efficient data handling


class BinaryClassifier(nn.Module):
    def __init__(self, input_shape):
        super(BinaryClassifier, self).__init__()
        # First layer with input size "input_shape" and output size 64
        self.layer1 = nn.Linear(input_shape, 64)
        # Activation function to introduce non-linearity
        self.relu = nn.ReLU()
        # Dropout to prevent overfitting by randomly setting input units to 0 with a probability of 0.5 during training
        self.dropout = nn.Dropout(0.5)
        # Output layer that maps from the hidden layer to a single output
        self.output = nn.Linear(64, 1)
        # Sigmoid activation function to output probabilities for the binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Forward pass through the network layers
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x


# Function to train the model
def train_model(
    model, criterion, optimizer, train_loader, val_loader, epochs=100, patience=10
):
    best_val_loss = float("inf")
    patience_counter = 0  # Counter for early stopping

    # Training loop
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        # Loop over each batch from the training set
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Clear previous gradients
            outputs = model(inputs)  # Compute model output
            loss = criterion(outputs, labels.view(-1, 1).float())  # Calculate loss
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update model parameters
            running_loss += loss.item()

        # Validation loop
        val_loss = 0.0
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels.view(-1, 1).float())
                val_loss += loss.item()

        val_loss /= len(val_loader)  # Average validation loss
        # print( f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.2f}, Validation Loss: {val_loss:.2f}")

        # Check for validation loss improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(), "models/titanic_pytorch_model.pth"
            )  # Save model
            patience_counter = 0
        else:
            patience_counter += 1  # Increment patience counter if no improvement

        # Early stopping condition
        if patience_counter >= patience:
            # print("Early stopping triggered.")
            break


# Main function to run the training process
def pytorch_main(X_train, y_train, X_val, y_val, return_scaler=True):
    # Scale features
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Convert scaled data into PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.float32)

    # Create DataLoader instances for batch processing and shuffling
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Initialize model, loss function, and optimizer
    model = BinaryClassifier(X_train_scaled.shape[1])
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
    optimizer = optim.Adam(model.parameters())  # Adam optimizer

    # Train the model
    train_model(model, criterion, optimizer, train_loader, val_loader)

    # Load the best model and evaluate on validation set
    model.load_state_dict(torch.load("models/titanic_pytorch_model.pth"))
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation for inference
        predictions_proba = model(X_val_tensor)
        # Convert probabilities to binary predictions based on a threshold of 0.5
        predictions = (predictions_proba.squeeze() > 0.5).int()

    # Optionally return the scaler along with the model and predictions
    if return_scaler:
        return model, predictions.numpy(), scaler
    else:
        return model, predictions.numpy()
