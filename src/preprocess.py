import pandas as pd
from sklearn.impute import SimpleImputer

from src.features import apply_feature_engineering


def preprocess_data(train_data, test_data):
    """
    Prepare the datasets for training/testing by selecting relevant features,
    handling missing values, converting categorical data to numerical,
    and ensuring train and test data have the same set of features.

    Args:
        train_data (DataFrame): The training dataset.
        test_data (DataFrame): The testing dataset.

    Returns:
        Tuple[DataFrame, DataFrame]: The processed training and testing datasets.
    """

    # Apply feature engineering
    train_data = apply_feature_engineering(train_data)
    test_data = apply_feature_engineering(test_data)

    # Define numeric and categorical features
    numeric_features = [
        "Pclass",
        "SibSp",
        "Parch",
        "Fare",
        "Age",
        "FamilySize",
        "IsAlone",
        "HasCabin",
    ]
    categorical_features = ["Sex", "Embarked", "Title", "AgeBin", "FareBin"]

    # Impute missing values in numeric data
    imputer = SimpleImputer(strategy="median")
    train_numeric_data = pd.DataFrame(
        imputer.fit_transform(train_data[numeric_features]), columns=numeric_features
    )
    test_numeric_data = pd.DataFrame(
        imputer.transform(test_data[numeric_features]), columns=numeric_features
    )

    # One-hot encode the categorical data
    combined_categorical_data = pd.get_dummies(
        pd.concat(
            [train_data[categorical_features], test_data[categorical_features]], axis=0
        )
    )
    # Split back into the training and test sets
    train_categorical_data = combined_categorical_data[: len(train_data)]
    test_categorical_data = combined_categorical_data[len(train_data) :]

    # Combine the numeric and categorical data
    X_train = pd.concat([train_numeric_data, train_categorical_data], axis=1)
    X_test = pd.concat([test_numeric_data, test_categorical_data], axis=1)

    return X_train, X_test
