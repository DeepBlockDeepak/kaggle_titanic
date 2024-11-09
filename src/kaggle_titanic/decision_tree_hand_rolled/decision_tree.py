from abc import ABC, abstractmethod
from collections import Counter
from typing import Union


class Node(ABC):
    def __init__(self, value) -> None:
        self.value = value  # the feature value leading to this node.
    
    @abstractmethod
    def predict(self, test_datum=None):
        pass


class Leaf(Node):
    """
    Represents a final decision point in the decision tree.
    Contains the labels of the training data that have reached this point.
    """
    def __init__(self, labels, value):
        super().__init__(value)
        self.labels = Counter(labels)

    def predict(self, test_datum=None) -> int:
        """
        Returns the most common label in this leaf.
        This is the prediction for any data point that reaches this leaf.
        
        Args:
            test_datum: Never used for the Leaf node, only InternalNodes

        Returns:
            int : The feature index of the most common label.
        """
        # return max(self.labels, key=self.labels.get)
        return self.labels.most_common(1)[0][0]


class InternalNode(Node):
    """
    Represents a decision point where the dataset is split based on the value of a particular feature.
    """
    def __init__(self, feature, branches, value):
        super().__init__(value)
        self.feature: int = feature  # The column index of the feature this node splits on.
        self.branches: list[Union["Leaf", "InternalNode"]] = branches  # Child nodes, which can be either further InternalNodes or Leafs.

    def predict(self, test_datum) -> int:
        """
        Determines the branch to follow based on the test_datum's feature value.
        When making a prediction, this node examines the value of its splitting
        feature in the input datapoint and selects the corresponding branch.
        
        Args:
            test_datum: datum from the testing data

        Returns:
            int : The feature index of the most common label.
        """
        test_feature_value = test_datum[self.feature]  # extract the feature value
        for branch in self.branches:
            if branch.value == test_feature_value:  # pursue the node in the tree
                return branch.predict(test_datum)

        # when no matching branch is found
        # aggregate labels from all reachable leaves and predict the most common label.
        all_leaf_labels = []

        def collect_leaf_labels(node: Union["Leaf", "InternalNode"]):
            if isinstance(node, Leaf):
                all_leaf_labels.extend(node.labels.elements())
            else:
                for branch in node.branches:
                    collect_leaf_labels(branch)

        collect_leaf_labels(self)
        
        # further error handling needed for if tree-creation is bad (no leaves)
        if not all_leaf_labels:
            raise ValueError("No labels found in the subtree.")

        return Counter(all_leaf_labels).most_common(1)[0][0]


class DecisionTree:
    """
    Enapsualtes the Tree model and prediction methods. 
    """

    def __init__(self) -> None:
        self.root: Union["InternalNode", "Leaf"] = None

    def fit(self, data: list[list], labels: list, value="") -> None:
        self.root = self._build_tree(data, labels, value)

    def predict(self, test_data: list[list]) -> list[int]:
        return [self.root.predict(datum) for datum in test_data]

    def _split(self, dataset, labels, column):
        # initialize lists to hold subsets of data and labels after splitting
        data_subsets = []
        label_subsets = []
        # extract unique values from the specified column to determine split points
        unique_vals = list(set([data[column] for data in dataset]))
        # Sort the unique values for consistent splitting
        unique_vals.sort()

        # for each unique value, create a subset of data and labels where the data matches the unique value
        for k in unique_vals:
            new_data_subset = [data for data in dataset if data[column] == k]
            new_label_subset = [
                labels[i] for i, data in enumerate(dataset) if data[column] == k
            ]

            # add the subsets to their respective lists
            data_subsets.append(new_data_subset)
            label_subsets.append(new_label_subset)

        # return the subsets of data and labels after splitting
        return data_subsets, label_subsets

    def _gini(self, dataset):
        # count the occurrences of each label in the dataset
        label_counts = Counter(dataset)
        # Gini impurity formula:
        impurity = 1 - sum(
            (l_count / len(dataset)) ** 2 for l_count in label_counts.values()
        )
        # gini impurity quantifies the disorder of a set; 0 represents complete purity
        return impurity

    def _information_gain(self, starting_labels, split_labels):
        # calculate the initial impurity before the split
        info_gain = self._gini(starting_labels)
        # calculate weighted impurity of each split
        weighted_impurity = sum(
            (len(subset) / len(starting_labels)) * self._gini(subset) for subset in split_labels
        )

        # information gain is the reduction in impurity after the split
        info_gain -= weighted_impurity

        return info_gain

    def _find_best_split(self, dataset, labels):
        # initialize variables to track the best split
        best_feature, best_gain = 0, 0
        num_features = len(dataset[0])

        # iterate over each feature in the dataset
        for feature in range(num_features):
            # split the dataset by the current feature
            data_subsets, label_subsets = self._split(dataset, labels, feature)
            # calculate the information gain from this split
            gain = self._information_gain(labels, label_subsets)
            # better information gain -> update best split
            if gain > best_gain:
                best_feature, best_gain = feature, gain

        return best_feature, best_gain

    def _build_tree(self, data: list, labels: list[int], value=""):
        # feature to split on for the current dataset
        best_feature, best_gain = self._find_best_split(data, labels)
        # no information gain
        if best_gain == 0:
            return Leaf(Counter(labels), value)

        # split the dataset based on the best feature and recursively build the tree.
        data_subsets, label_subsets = self._split(data, labels, best_feature)

        branches = []
        for i in range(len(data_subsets)):
            branch = self._build_tree(
                data_subsets[i], label_subsets[i], data_subsets[i][0][best_feature]
            )
            branches.append(branch)

        # InternalNode that splits the data at the best feature.
        return InternalNode(best_feature, branches, value)
