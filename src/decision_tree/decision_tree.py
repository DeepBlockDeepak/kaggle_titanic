from collections import Counter
from typing import List, Union


# A Leaf node represents a final decision point in the decision tree.
# It contains the labels of the training data that have reached this point.
class Leaf:
    def __init__(self, labels, value):
        self.labels = Counter(
            labels
        )  # Counter of labels for the data points that reach this leaf.
        self.value = value  # Represents the feature value leading to this leaf in the parent node.

    # Returns the most common label in this leaf. This is the prediction
    # for any data point that reaches this leaf.
    def predict(self, _):
        # return max(self.labels, key=self.labels.get)
        return self.labels.most_common(1)[0][0]


# An Internal_Node represents a decision point where the dataset is split
# based on the value of a particular feature.
class Internal_Node:
    def __init__(self, feature, branches, value):
        self.feature: int = (
            feature  # The column index of the feature this node splits on.
        )
        self.branches: List[
            Union["Leaf", "Internal_Node"]
        ] = branches  # Child nodes, which can be either further Internal_Nodes or Leafs.
        self.value = value  # Represents the feature value that leads to this node in the parent node.

    # Determines the branch to follow based on the test data's feature value.
    # When making a prediction, the node examines the value of its splitting
    # feature in the input datapoint and selects the corresponding branch.
    def predict(self, test_datapoint):
        test_feature_value = test_datapoint[self.feature]
        for branch in self.branches:
            if branch.value == test_feature_value:
                return branch.predict(test_datapoint)

        # if no matching branch is found
        # Fallback: Aggregate labels from all reachable leaves and predict the most common label.
        all_leaf_labels = []

        def collect_leaf_labels(node: Union["Leaf", "Internal_Node"]):
            if isinstance(node, Leaf):
                all_leaf_labels.extend(node.labels.elements())
            else:
                for branch in node.branches:
                    collect_leaf_labels(branch)

        collect_leaf_labels(self)
        return Counter(all_leaf_labels).most_common(1)[0][0]


def split(dataset, labels, column):
    data_subsets = []
    label_subsets = []
    unique_vals = list(set([data[column] for data in dataset]))
    unique_vals.sort()
    for k in unique_vals:
        new_data_subset = [data for data in dataset if data[column] == k]
        new_label_subset = [
            labels[i] for i, data in enumerate(dataset) if data[column] == k
        ]

        data_subsets.append(new_data_subset)
        label_subsets.append(new_label_subset)

    return data_subsets, label_subsets


def gini(dataset):
    label_counts = Counter(dataset)
    impurity = 1 - sum(
        (l_count / len(dataset)) ** 2 for l_count in label_counts.values()
    )
    return impurity


def information_gain(starting_labels, split_labels):
    info_gain = gini(starting_labels)
    weighted_impurity = sum(
        (len(subset) / len(starting_labels)) * gini(subset) for subset in split_labels
    )

    info_gain -= weighted_impurity

    return info_gain


def find_best_split(dataset, labels):
    best_feature, best_gain = 0, 0
    for feature in range(len(dataset[0])):
        data_subsets, label_subsets = split(dataset, labels, feature)
        gain = information_gain(labels, label_subsets)
        if gain > best_gain:
            best_feature, best_gain = feature, gain

    return best_feature, best_gain


def build_tree(data, labels, value=""):
    # Find the best feature to split on for the current dataset
    best_feature, best_gain = find_best_split(data, labels)
    # No information gain
    if best_gain == 0:
        return Leaf(Counter(labels), value)

    # Split the dataset based on the best feature and recursively build the tree.
    data_subsets, label_subsets = split(data, labels, best_feature)

    branches = []
    for i in range(len(data_subsets)):
        branch = build_tree(
            data_subsets[i], label_subsets[i], data_subsets[i][0][best_feature]
        )
        branches.append(branch)

    # Return an Internal_Node that splits the data at the best feature.
    return Internal_Node(best_feature, branches, value)
