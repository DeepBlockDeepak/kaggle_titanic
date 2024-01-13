# commenting out until I have resolved path import design & issue here
# Current output is: "Decision Tree Classification Accuracy: 0.72"
# import sys
# sys.path.append('/path/to/kaggle_titanic/root')

from src.preprocess import preprocess_data
from decision_tree import build_tree, classify 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


def debug_accuracy_score(y_val_list, predictions):
    # Check lengths
    if len(y_val_list) != len(predictions):
        print("Mismatch in lengths")
        print("Length of y_val_list:", len(y_val_list))
        print("Length of predictions:", len(predictions))
        return

    # Check types
    y_val_types = set(type(item) for item in y_val_list)
    pred_types = set(type(item) for item in predictions)

    if len(y_val_types) > 1 or len(pred_types) > 1 or y_val_types != pred_types:
        print("Mismatch in data types")
        print("Types in y_val_list:", y_val_types)
        print("Types in predictions:", pred_types)
        return

    # Check for NaN, None, or non-numeric values
    if any(x is None or np.isnan(x) for x in y_val_list + predictions):
        print("Found None or NaN values")
        return

    # Print samples
    print("Sample of true labels:", y_val_list[:10])
    print("Sample of predictions:", predictions[:10])

    # Check unique values
    print("Unique values in y_val_list:", set(y_val_list))
    print("Unique values in predictions:", set(predictions))

    # Attempt to calculate accuracy
    try:
        accuracy = accuracy_score(y_val_list, predictions)
        print("Accuracy:", accuracy)
    except ValueError as e:
        print("Error calculating accuracy:", e)


def decision_tree_main():
    # Load and preprocess data
    train_data = pd.read_csv('/Users/jmeds/code/kaggle_titanic/data/train.csv')
    test_data = pd.read_csv('/Users/jmeds/code/kaggle_titanic/data/test.csv')
    X_train, X_test = preprocess_data(train_data, test_data)
    y_train = train_data["Survived"]

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

    # Convert DataFrame to list of lists for the decision tree compatibility
    X_train_list = X_train.values.tolist()
    X_val_list = X_val.values.tolist()
    y_train_list = y_train.tolist()
    y_val_list = y_val.tolist()

    # Build and train decision tree
    tree = build_tree(X_train_list, y_train_list)

    # Score the Decision Tree with predictions on the validation set
    predictions = [classify(val_data, tree) for val_data in X_val_list]
    print(f"Decision Tree Classification Accuracy: {accuracy_score(y_val_list, predictions):.2f}")

    # debug_accuracy_score(y_val_list, predictions)

if __name__ == "__main__":
    decision_tree_main()
