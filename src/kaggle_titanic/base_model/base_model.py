# base_model.py
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score

class BaseModel(ABC):
    def __init__(self):
        self.model = None
        self.name = 'ABC Model'

    @abstractmethod
    def train(self, X_train, y_train, X_val, y_val):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def evaluate(self, X_val, y_val):
        predictions = self.predict(X_val)
        accuracy = accuracy_score(y_val, predictions)
        print(f"Model: {self.name}, Accuracy: {accuracy:.2f}")
        return predictions
