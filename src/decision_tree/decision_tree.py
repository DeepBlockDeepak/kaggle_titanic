from collections import Counter

class Leaf:
    def __init__(self, labels, value):
        self.labels = Counter(labels)
        self.value = value

    def predict(self, _):
        return max(self.labels, key=self.labels.get)


class Internal_Node:
    def __init__(self, feature, branches, value):
        self.feature = feature
        self.branches = branches
        self.value = value

    def predict(self, datapoint):
        value = datapoint[self.feature]
        for branch in self.branches:
            if branch.value == value:
                return branch.predict(datapoint)
        # Fallback: if no matching branch is found
        all_leaf_labels = []
        def collect_leaf_labels(node):
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
        new_label_subset = [labels[i] for i, data in enumerate(dataset) if data[column] == k]

        data_subsets.append(new_data_subset)
        label_subsets.append(new_label_subset)
    
    return data_subsets, label_subsets

def gini(dataset):
    impurity = 1
    label_counts = Counter(dataset)
    impurity = 1 - sum((l_count/len(dataset))**2 for l_count in label_counts.values())
    return impurity


def information_gain(starting_labels, split_labels):
    info_gain = gini(starting_labels)
    weighted_impurity = sum((len(subset) / len(starting_labels)) * gini(subset) for subset in split_labels)

    info_gain -= weighted_impurity
    
    return info_gain

def find_best_split(dataset, labels):
    best_gain, best_feature = 0, 0
    for feature in range(len(dataset[0])):
        data_subsets, label_subsets = split(dataset, labels, feature)
        gain = information_gain(labels, label_subsets)
        if gain > best_gain:
            best_gain, best_feature = gain, feature
    
    return best_feature, best_gain

def build_tree(data, labels, value = ""):
    best_feature, best_gain = find_best_split(data, labels)
    if best_gain == 0:
        return Leaf(Counter(labels), value)
    data_subsets, label_subsets = split(data, labels, best_feature)
    branches = []
    for i in range(len(data_subsets)):
        branch = build_tree(data_subsets[i], label_subsets[i], data_subsets[i][0][best_feature])
        branches.append(branch)

    return Internal_Node(best_feature, branches, value)
          
def classify(datapoint, tree):
    
    if isinstance(tree, Leaf):
        return max(tree.labels.items(), key=lambda item: item[1])[0]

    # else the tree is an Internal_Node 
    value = datapoint[tree.feature]
    for branch in tree.branches:
        if branch.value == value:
            return classify(datapoint, branch)

    # If no matching branch is found, return the most common label
    # Aggregate labels from all leaves
    all_leaf_labels = []
    def collect_leaf_labels(node):
        if isinstance(node, Leaf):
            all_leaf_labels.extend(node.labels.elements())
        else:
            for branch in node.branches:
                collect_leaf_labels(branch)

    collect_leaf_labels(tree)
    # Return the most common label
    return Counter(all_leaf_labels).most_common(1)[0][0]