import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from kaggle_titanic.features import apply_feature_engineering


def preprocess_data(train_data: pd.DataFrame, test_data: pd.DataFrame):
    """
    Prepare the datasets for training/testing by selecting relevant features,
    handling missing values, converting categorical data to numerical,
    and ensuring train and test data have the same set of features.

    Args:
        train_data: The training dataset.
        test_data: The testing dataset.

    Returns:
        Tuple[DataFrame, DataFrame]: The processed training and testing datasets.
    """

    # apply feature engineering
    train_data = apply_feature_engineering(train_data)
    test_data = apply_feature_engineering(test_data)

    # handling missing values in 'Embarked'
    train_data["Embarked"] = train_data["Embarked"].fillna(
        train_data["Embarked"].mode()[0]
    )
    test_data["Embarked"] = test_data["Embarked"].fillna(
        test_data["Embarked"].mode()[0]
    )

    # hand-define numeric and categorical features
    numeric_features = [
        "Pclass",
        "SibSp",
        "Parch",
        "Fare",
        "FamilySize",
        "IsAlone",  # consider dropping bool-int col from scaling
        "HasCabin",  # consider dropping bool-int col from scaling
    ]
    categorical_features = ["Sex", "Embarked", "Title", "AgeBin", "FareBin"]

    # impute missing values in numeric data
    imputer = SimpleImputer(strategy="median")
    train_numeric_data = pd.DataFrame(
        imputer.fit_transform(train_data[numeric_features]), columns=numeric_features
    )
    test_numeric_data = pd.DataFrame(
        imputer.transform(test_data[numeric_features]), columns=numeric_features
    )

    # # convert boolean features to integers
    # boolean_features = ["IsAlone", "HasCabin"]
    # train_numeric_data[boolean_features] = train_numeric_data[boolean_features].astype(int)
    # test_numeric_data[boolean_features] = test_numeric_data[boolean_features].astype(int)

    # scale numeric data
    scaler = StandardScaler()
    train_numeric_data = pd.DataFrame(
        scaler.fit_transform(train_numeric_data), columns=numeric_features
    )
    test_numeric_data = pd.DataFrame(
        scaler.transform(test_numeric_data), columns=numeric_features
    )

    # one-hot encode the categorical data
    combined_categorical_data = pd.get_dummies(
        pd.concat(
            [train_data[categorical_features], test_data[categorical_features]], axis=0
        )
    )

    # convert boolean columns to integers before splitting
    bool_cols = combined_categorical_data.select_dtypes(include="bool").columns
    combined_categorical_data[bool_cols] = combined_categorical_data[bool_cols].astype(
        int
    )

    # split back into the training and test sets
    train_categorical_data = combined_categorical_data[: len(train_data)]
    test_categorical_data = combined_categorical_data[len(train_data) :]

    # combine the numeric and categorical data
    X_train = pd.concat([train_numeric_data, train_categorical_data], axis=1)
    X_test = pd.concat([test_numeric_data, test_categorical_data], axis=1)

    return X_train, X_test
