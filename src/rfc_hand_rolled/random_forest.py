import random
from collections import Counter

from sklearn.metrics import accuracy_score

from src.decision_tree.decision_tree import Internal_Node, Leaf, information_gain, split


class RandomForestClassifier:
    def __init__(self, n_trees=10, n_features=None):
        self.n_trees = n_trees
        self.n_features = n_features
        self.trees = []

    def fit(self, dataset, labels):
        self.trees = []
        for _ in range(self.n_trees):
            # Generate a bootstrap sample
            boot_dataset, boot_labels = bootstrap_sample(dataset, labels)
            # Determine the number of features to consider if not specified
            if self.n_features is None:
                self.n_features = int(len(dataset[0]) ** 0.5)
            # Build a decision tree on the bootstrap sample
            tree = build_tree_random_features(
                boot_dataset, boot_labels, self.n_features
            )
            self.trees.append(tree)

    def predict(self, dataset):
        predictions = []
        for data_point in dataset:
            # Collect predictions from all trees
            votes = [tree.predict(data_point) for tree in self.trees]
            # Majority vote
            predictions.append(Counter(votes).most_common(1)[0][0])
        return predictions


def bootstrap_sample(dataset, labels):
    n_samples = len(dataset)
    # Generate random indices with replacement
    indices = [random.randint(0, n_samples - 1) for _ in range(n_samples)]
    # Create the bootstrap sample
    bootstrap_dataset = [dataset[i] for i in indices]
    bootstrap_labels = [labels[i] for i in indices]

    return bootstrap_dataset, bootstrap_labels


def find_best_split_random_features(dataset, labels, n_features):
    best_feature, best_gain = -1, 0
    dataset_features = list(range(len(dataset[0])))
    # Randomly select n_features to consider for splitting
    features_to_consider = random.sample(dataset_features, n_features)

    for feature in features_to_consider:
        data_subsets, label_subsets = split(dataset, labels, feature)
        gain = information_gain(labels, label_subsets)
        if gain > best_gain:
            best_feature, best_gain = feature, gain

    return best_feature, best_gain


# Update the build_tree function to use find_best_split_random_features
def build_tree_random_features(data, labels, n_features, value=""):
    # Find the best feature to split on for the current dataset, considering a random subset of features
    best_feature, best_gain = find_best_split_random_features(data, labels, n_features)
    if best_gain == 0:
        return Leaf(Counter(labels), value)

    data_subsets, label_subsets = split(data, labels, best_feature)
    branches = []
    for i in range(len(data_subsets)):
        branch = build_tree_random_features(
            data_subsets[i],
            label_subsets[i],
            n_features,
            data_subsets[i][0][best_feature],
        )
        branches.append(branch)

    return Internal_Node(best_feature, branches, value)


def rfc_handroll_main(X_train, y_train, X_val, y_val):
    # Convert DataFrame to list of lists for the decision tree compatibility
    X_train_list = X_train.values.tolist()
    X_val_list = X_val.values.tolist()
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
        f"Model: RandomForestClassifier Hand-Rolled, Accuracy: {accuracy_score(y_val_list, predictions):.2f}"
    )

    return rfc, predictions
