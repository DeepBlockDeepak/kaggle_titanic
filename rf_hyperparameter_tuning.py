import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split

from src.preprocess import preprocess_data

""" RESULTS:
Best parameters: {'bootstrap': True, 'criterion': 'entropy', 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50}
Best cross-validation score: 0.8455727371220328
Validation accuracy of the best model: 0.7877094972067039
"""


def perform_grid_search(X_train, y_train):
    # Define the parameter grid to search
    param_grid = {
        "n_estimators": [50, 100, 200, 300, 500],  # Number of trees in the forest.
        "max_depth": [
            5,
            10,
            15,
            20,
            25,
            None,
        ],  # Maximum depth of the tree. If None, nodes are expanded until all leaves are pure.
        "min_samples_split": [
            2,
            5,
            10,
            15,
        ],  # Minimum number of samples required to split an internal node.
        "min_samples_leaf": [
            1,
            2,
            4,
            6,
        ],  # Minimum number of samples required to be at a leaf node.
        "bootstrap": [
            True,
            False,
        ],  # Whether bootstrap samples are used when building trees.
        "max_features": [
            "auto",
            "sqrt",
            "log2",
        ],  # Number of features to consider when looking for the best split.
        "criterion": ["gini", "entropy"],  # Function to measure the quality of a split.
    }

    # Initialize the RandomForestClassifier
    rf_model = RandomForestClassifier(random_state=1)

    # Initialize the GridSearchCV with 5-fold cross-validation
    grid_search = GridSearchCV(
        estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2
    )

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Print the best parameters and the best score
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_}")

    return grid_search.best_estimator_


if __name__ == "__main__":
    # Load training data
    train_data = pd.read_csv("data/train.csv")

    # Process data
    X_train, _ = preprocess_data(
        train_data, train_data
    )  # Preprocess train_data twice since test_data is not needed here
    y_train = train_data["Survived"]

    # Split data into training and validation sets
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=1
    )

    # Perform grid search
    best_model = perform_grid_search(X_train_split, y_train_split)

    # Validate the best model
    val_predictions = best_model.predict(X_val_split)
    val_accuracy = accuracy_score(y_val_split, val_predictions)
    print(f"Validation accuracy of the best model: {val_accuracy}")
