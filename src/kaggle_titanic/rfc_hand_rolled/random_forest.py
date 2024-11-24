import random
from collections import Counter

from sklearn.metrics import accuracy_score

from kaggle_titanic.decision_tree_hand_rolled.decision_tree import DecisionTree


class RandomDecisionTree(DecisionTree):
    def __init__(self, n_features=None):
        super().__init__()
        self.n_features = n_features

    # overload the DecisionTree method to obtain random features/Trees
    def _find_best_split(self, dataset, labels):
        best_feature, best_gain = -1, 0
        num_features = len(dataset[0])
        dataset_features = list(range(num_features))

        # determine the number of features to consider
        if self.n_features is None:
            n_features_to_consider = int(num_features**0.5)
        else:
            n_features_to_consider = self.n_features

        # randomly select n_features to consider for splitting
        features_to_consider = random.sample(dataset_features, n_features_to_consider)

        for feature in features_to_consider:
            data_subsets, label_subsets = self._split(dataset, labels, feature)
            gain = self._information_gain(labels, label_subsets)
            if gain > best_gain:
                best_feature, best_gain = feature, gain

        return best_feature, best_gain


class RandomForestClassifier:
    def __init__(self, n_trees=10, n_features=None):
        self.n_trees = n_trees
        self.n_features = n_features
        self.trees: list[RandomDecisionTree] = []

    def fit(self, dataset, labels):
        for _ in range(self.n_trees):
            # generate a bootstrap sample
            boot_dataset, boot_labels = self._bootstrap_sample(dataset, labels)

            tree = RandomDecisionTree(n_features=self.n_features)
            tree.fit(boot_dataset, boot_labels)
            self.trees.append(tree)

    def predict(self, dataset):
        predictions = []
        for sample in dataset:
            # collect predictions from all trees
            votes = [tree.root.predict(sample) for tree in self.trees]
            # majority vote
            predictions.append(Counter(votes).most_common(1)[0][0])
        return predictions

    def _bootstrap_sample(self, dataset, labels):
        n_samples = len(dataset)
        # generate random indices with replacement
        indices = [random.randint(0, n_samples - 1) for _ in range(n_samples)]
        # create the bootstrap sample
        bootstrap_dataset = [dataset[i] for i in indices]
        bootstrap_labels = [labels[i] for i in indices]

        return bootstrap_dataset, bootstrap_labels


def rfc_handroll_main(X_train, y_train, X_val, y_val):
    # Convert DataFrame to list of lists for the decision tree compatibility
    X_train_list = X_train
    X_val_list = X_val
    y_train_list = y_train.tolist()
    y_val_list = y_val.tolist()

    # Instantiate the RandomForestClassifier
    rfc = RandomForestClassifier(
        n_trees=10, n_features=4
    )  # Example: 10 trees, each considering 4 features

    # Fit the RandomForestClassifier on the training data
    rfc.fit(X_train_list, y_train_list)

    # Use the trained RandomForest to make predictions on the validation set
    predictions = rfc.predict(X_val_list)

    # evaluate and print the accuracy
    print(
        f"Model: Custom RandomForestClassifier, Accuracy: {accuracy_score(y_val_list, predictions):.2f}"
    )

    return rfc, predictions
