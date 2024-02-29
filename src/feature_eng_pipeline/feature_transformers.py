import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ExtractTitleTransformer(BaseEstimator, TransformerMixin):
    """
    Extracts the title from the 'Name' column and creates a new 'Title' column.

    'the regex captures any word immediately followed by
    a period character -> the title in the names.
    """

    def fit(self, X, y=None):
        return self  # nothing to fit

    def transform(self, X, y=None):
        X = X.copy()  # avoid changing the original dataset
        X["Title"] = X["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)

        return X


class CreateFamilySizeTransformer(BaseEstimator, TransformerMixin):
    """
    New feature adds 'SibSp' and 'Parch' columns plus one for the passenger themself.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X["FamilySize"] = X["SibSp"] + X["Parch"] + 1

        return X


class CreateIsAloneTransformer(BaseEstimator, TransformerMixin):
    """
    new feature is 1 if FamilySize is 1, else 0.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X["IsAlone"] = (X["FamilySize"] == 1).astype(int)

        return X


class CreateCabinIndicatorTransformer(BaseEstimator, TransformerMixin):
    """
    New binary 'HasCabin' feature. 1 if 'Cabin' data is available (not NaN), else 0.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X["HasCabin"] = X["Cabin"].apply(lambda x: 0 if pd.isna(x) else 1)

        return X


class CreateAgeBinsTransformer(BaseEstimator, TransformerMixin):
    """
    Categorizes 'Age' into bins and creates a new 'AgeBin' column with the category labels.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        bins = [0, 12, 20, 40, 60, 80]
        labels = ["Child", "Teen", "Adult", "MiddleAged", "Senior"]
        X["AgeBin"] = pd.cut(X["Age"], bins, labels=labels)

        return X


class CreateFareBinsTransformer(BaseEstimator, TransformerMixin):
    """
    Categorizes 'Fare' into bins and creates a new 'FareBin' column with the category labels.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        bins = [0, 7.91, 14.454, 31, 512]
        labels = ["Low", "Below_Average", "Above_Average", "High"]
        X["FareBin"] = pd.cut(X["Fare"], bins, labels=labels)

        return X
