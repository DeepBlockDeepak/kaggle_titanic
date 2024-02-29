from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_eng_pipeline.feature_transformers import (
    CreateAgeBinsTransformer,
    CreateCabinIndicatorTransformer,
    CreateFamilySizeTransformer,
    CreateFareBinsTransformer,
    CreateIsAloneTransformer,
    ExtractTitleTransformer,
)

# Custom feature engineering pipeline
feature_engineering_pipeline = Pipeline(
    [
        ("extract_title", ExtractTitleTransformer()),
        ("create_family_size", CreateFamilySizeTransformer()),
        ("create_is_alone", CreateIsAloneTransformer()),
        ("create_cabin_indicator", CreateCabinIndicatorTransformer()),
        ("create_age_bins", CreateAgeBinsTransformer()),
        ("create_fare_bins", CreateFareBinsTransformer()),
    ]
)


# Preprocessing pipeline for numeric and categorical features
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

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Combine everything into a large preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="passthrough",  # To allow the passage of new features created
)

# Full pipeline with feature engineering then preprocessing
full_pipeline = Pipeline(
    steps=[
        ("feature_engineering", feature_engineering_pipeline),
        ("preprocessing", preprocessor),
    ]
)


def preprocess_data(train_data, test_data):
    # Apply the full pipeline to both training and test data
    X_train = full_pipeline.fit_transform(train_data)
    X_test = full_pipeline.transform(test_data)

    return X_train, X_test
