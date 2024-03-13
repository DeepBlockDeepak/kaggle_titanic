import numpy as np
from sklearn.metrics import accuracy_score

from kaggle_titanic.decision_tree_hand_rolled.decision_tree import build_tree


def custom_accuracy_score(true_labels, predictions):
    correct = sum(1 for true, pred in zip(true_labels, predictions) if true == pred)
    return correct / len(true_labels)


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
        print(f"Accuracy: {accuracy:.2f}")
    except ValueError as e:
        print("Error calculating accuracy:", e)


def decision_tree_main(X_train, y_train, X_val, y_val):
    # Convert DataFrame to list of lists for the decision tree compatibility
    X_train_list = X_train.values.tolist()
    X_val_list = X_val.values.tolist()
    y_train_list = y_train.tolist()
    y_val_list = y_val.tolist()

    # Build and train decision tree
    tree = build_tree(X_train_list, y_train_list)

    # Score the Decision Tree with predictions on the validation set
    predictions = [tree.predict(val_data) for val_data in X_val_list]
    # debug_accuracy_score(y_val_list, predictions)
    print(
        f"Model: Custom Decision Tree, Accuracy: {accuracy_score(y_val_list, predictions):.2f}"
    )

    return tree, predictions
