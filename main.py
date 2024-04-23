import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from kaggle_titanic.decision_tree_hand_rolled.decision_tree_train import (
    decision_tree_main,
)
from kaggle_titanic.my_neural_network.simple_nn_main import simple_nn_main
from kaggle_titanic.naive_bayes.bayes_main import naive_bayes_main
from kaggle_titanic.preprocess import preprocess_data
from kaggle_titanic.pytorch.pytorch_binary import pytorch_main
from kaggle_titanic.random_forest_classifier.rf_main import random_forest_main
from kaggle_titanic.rfc_hand_rolled.random_forest import rfc_handroll_main
from kaggle_titanic.svm.svm_train import svm_main
from kaggle_titanic.visualize import (
    plot_confusion_matrix,
    plot_feature_importances,
    plot_model_accuracies,
    plot_roc_curve,
    plot_survival_probability_histogram,
)


def create_submission_file(test_data, test_predictions):
    output = pd.DataFrame(
        {"PassengerId": test_data.PassengerId, "Survived": test_predictions}
    )
    output.to_csv("submission.csv", index=False)
    print("Your submission was successfully saved!")


def parse_args():
    parser = argparse.ArgumentParser(
        description="""
        Titanic Survival Prediction using different models.
        - Random Forest: A versatile machine learning method capable of performing both regression and classification tasks.
        - Decision Tree: A hand-rolled decision tree implementation for simple yet effective classification modeling.
        - SVM (Support Vector Machine): Effective for high-dimensional spaces.
        Select the model via the --model argument.
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=[
            "random_forest",
            "svm",
            "decision_tree",
            "custom_rfc",
            "naive_bayes",
            "pytorch",
            "simple_nn",
            "all",
        ],
        help='Specify the model to train or use "all" to compare models.',
    )
    return parser.parse_args()


# Train and evaluate models based with their respective main scripts
def train_and_score(model_func, X_train, y_train, X_val, y_val):
    model, predictions = model_func(X_train, y_train, X_val, y_val)
    return accuracy_score(y_val, predictions)


def main():
    args = parse_args()

    # Load data
    try:
        train_data = pd.read_csv("data/train.csv")
        test_data = pd.read_csv("data/test.csv")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(
            "Make sure train.csv and test.csv are in the 'data' directory and try again."
        )
        return

    # Process data
    X_train, X_test = preprocess_data(train_data, test_data)
    y_train = train_data["Survived"]
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=0
    )

    # Train and evaluate the RandomForestClassifier model
    if args.model == "random_forest":
        model, predictions = random_forest_main(X_train, y_train, X_val, y_val)
        test_predictions = model.predict(X_test)
    elif args.model == "decision_tree":
        # Convert X_test to list of lists for decision tree compatibility
        X_test_list = X_test.values.tolist()
        # Call the Decision Tree training and evaluation function
        model, predictions = decision_tree_main(X_train, y_train, X_val, y_val)
        test_predictions = [model.predict(test_data) for test_data in X_test_list]
    elif args.model == "custom_rfc":
        # Convert X_test to list of lists for decision tree compatibility
        X_test_list = X_test.values.tolist()
        # Call the Decision Tree training and evaluation function
        model, predictions = rfc_handroll_main(X_train, y_train, X_val, y_val)
        test_predictions = model.predict(X_test_list)
    elif args.model == "svm":
        # Call the SVM training and evaluation function
        model, predictions = svm_main(X_train, y_train, X_val, y_val)
        test_predictions = model.predict(X_test)
    elif args.model == "naive_bayes":
        # Call the Naive Bayes training and evaluation function
        model, predictions = naive_bayes_main(X_train, y_train, X_val, y_val)
        test_predictions = model.predict(X_test)
    elif args.model == "pytorch":
        model, predictions, scaler = pytorch_main(
            X_train, y_train, X_val, y_val, return_scaler=True
        )

        # Ensure X_test is preprocessed similarly to X_train and X_val
        X_test_scaled = scaler.transform(X_test)

        # Convert the scaled test data to a tensor
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

        # PyTorch model evaluation
        model.eval()  # set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation for inference
            test_predictions_proba = model(X_test_tensor)
            # Apply threshold to convert probabilities to binary class labels
            test_predictions = (
                (test_predictions_proba.squeeze() > 0.5).int().numpy().flatten()
            )
    elif args.model == "simple_nn":
        model, predictions = simple_nn_main(X_train, y_train, X_val, y_val)
        X_test = X_test.to_numpy().T  # Ensure test data is also prepared correctly
        X_test = X_test.astype(np.float64)  # Ensure consistent data type for predict()
        test_predictions = model.predict(X_test).flatten()
    elif args.model == "all":
        # Function Handler
        model_functions = {
            "RFC \n sklearn": random_forest_main,
            "SVM \n sklearn": svm_main,
            "Naive Bayes \n sklearn": naive_bayes_main,
            "Hand-Rolled \n DecTree": decision_tree_main,
            "Hand-Rolled \n RFC ": rfc_handroll_main,
            "PyTorch \n BinaryClassifier": lambda X_train, y_train, X_val, y_val: pytorch_main(
                X_train, y_train, X_val, y_val, return_scaler=False
            ),
        }

        # Initialize dictionary to store accuracy scores
        model_accuracies = {}
        # Iterate over models to train and evaluate
        for model_name, model_func in model_functions.items():
            accuracy = train_and_score(model_func, X_train, y_train, X_val, y_val)
            model_accuracies[model_name] = round(accuracy, 4)

        # Call the combined plot function with the accuracy scores
        plot_model_accuracies(model_accuracies)
        return  # subvert the individual plotting below
    else:
        print("No valid model selected. Defaulting to Random Forest.")
        return

    plot_feature_importances(model, X_train)
    plot_confusion_matrix(model, y_val, predictions)
    plot_roc_curve(model, X_val, y_val)
    plot_survival_probability_histogram(model, X_test)

    # Prepare the submission file which writes /submission.csv
    create_submission_file(test_data, test_predictions)


if __name__ == "__main__":
    main()
