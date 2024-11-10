# src/kaggle_titanic/base_model/base_model.py
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score

class BaseModel(ABC):
    def __init__(self, name='BaseModel'):
        self.model = None
        self.name = name

    @abstractmethod
    def fit(self, X_train, y_train):
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X):
        """Make predictions using the trained model."""
        pass

    def evaluate(self, X_val, y_val):
        """Evaluate the model and print accuracy."""
        predictions = self.predict(X_val)
        accuracy = accuracy_score(y_val, predictions)
        print(f"Model: {self.name}, Accuracy: {accuracy:.2f}")
        return predictions
