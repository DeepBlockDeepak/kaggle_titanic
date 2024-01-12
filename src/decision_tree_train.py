# src/decision_tree_train.py
from preprocess import preprocess_data
from decision_tree import build_tree, classify 
import pandas as pd

def train_decision_tree_model(X_train, y_train):
    # Train the decision tree model
    tree = build_tree(X_train, y_train)
    return tree

def decision_tree_main():
    # Load and process the data
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    X_train, X_test = preprocess_data(train_data, test_data)
    y_train = train_data["Survived"]
    
    # Train the decision tree model
    tree = train_decision_tree_model(X_train, y_train)
    
    # Here, you could evaluate the tree or use it to make predictions
    # For example:
    # predictions = [classify(row, tree) for row in X_train]
    # Evaluate predictions...

if __name__ == "__main__":
    decision_tree_main()
