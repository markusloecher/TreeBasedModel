import numpy as np
import pandas as pd
from collections import Counter
import copy

class Node:
    def __init__(self,
                 feature=None,
                 feature_name=None,
                 threshold=None,
                 left=None,
                 right=None,
                 gain=None,
                 id=None,
                 depth=None,
                 leaf_node=False,
                 samples=None,
                 gini=None,
                 *,
                 value=None,
                 clf_value_dis=None,
                 clf_prob_dis=None):
        self.feature = feature
        self.feature_name = feature_name
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain
        self.gini = gini
        self.value = value
        self.clf_value_dis = clf_value_dis
        self.clf_prob_dis = clf_prob_dis
        self.id = id
        self.depth = depth
        self.samples = samples
        self.leaf_node = leaf_node

    def is_leaf_node(self):
        return self.leaf_node is not False


class DecisionTree:
    def __init__(self,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_depth=100,
                 n_features=None,
                 criterion="gini",
                 treetype="classification",
                 k=None,
                 feature_names=None,
                 HShrinkage=False,
                 HS_lambda=0,
                 random_state=None):
        self.min_samples_split=min_samples_split
        self.min_samples_leaf = min_samples_leaf # Still need to be implemented
        self.max_depth=max_depth
        self.n_features=n_features
        self.feature_names = feature_names
        self.root=None
        self.criterion=criterion
        self.k=k
        self.treetype = treetype
        self.HShrinkage = HShrinkage
        self.HS_lambda = HS_lambda
        self.random_state = np.random.default_rng(random_state)
        self.n_nodes=0
        self.oob_preds=None #only for random forests


    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.node_list = []
        self.node_id_dict = {}
        self.no_samples_total = y.shape[0]

        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns
            X = X.values
        self.root = self._grow_tree(X, y, feature_names=self.feature_names)

        # Set decision paths and node ids as class attributes
        self._get_decision_paths()

        #Sort node_list by node id
        self.node_list = sorted(self.node_list, key=lambda o: o.id)

        #Apply Hierarchical Shrinkage (change value (prediction) in leaf
        if self.HShrinkage:
            self._apply_hierarchical_srinkage(treetype=self.treetype)

        # Create dict of dict for all nodes with important attributes per node
        for node in self.node_list:
            self.node_id_dict[node.id] = {
                "node": node,
                "id": node.id,
                "depth": node.depth,
                "feature": node.feature_name or node.feature,
                "is_leaf_node": node.leaf_node,
                "threshold": node.threshold,
                "gini": node.gini,
                "samples": node.samples,
                "value": node.value
            }
            if self.treetype=="classification":
                self.node_id_dict[
                    node.id]["value_distribution"] = node.clf_value_dis
                self.node_id_dict[
                    node.id]["prob_distribution"] = node.clf_prob_dis

        #Set max tree depth as class attributes
        depth_list = [len(i) for i in self.decision_paths]
        self.max_depth_ = max(depth_list)

        #Set feature_importances_ (aka information_gain_scaled) as class attribute
        self._get_feature_importance()


    def _grow_tree(self, X, y, depth=0, feature_names=None):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        #Calculate leaf value
        if self.treetype == "classification":
            #leaf_value = self._most_common_label(y)
            counter = Counter(y)
            clf_value_dis = [counter.get(0) or 0, counter.get(1) or 0]
            clf_prob_dis = (np.array(clf_value_dis) / n_samples)
            leaf_value = np.argmax(clf_prob_dis)

        elif self.treetype == "regression":
            leaf_value = self._mean_label(y)
            clf_value_dis = None
            clf_prob_dis = None

        # check the stopping criteria and creates leaf
        if (depth >= self.max_depth or n_labels == 1
                or n_samples < self.min_samples_split
                or n_samples < 2*self.min_samples_leaf):
            node = Node(value=leaf_value,
                        clf_value_dis=clf_value_dis,
                        clf_prob_dis = clf_prob_dis,
                        leaf_node=True,
                        gini=self._gini(y),
                        depth=depth,
                        samples=n_samples)
            self.node_list.append(node)
            return node

        feat_idxs = self.random_state.choice(n_feats, self.n_features, replace=False)

        # find the best split
        best_feature, best_thresh, best_gain = self._best_split(X, y, feat_idxs)

        # create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1, feature_names)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1, feature_names)

        best_feature_name=None
        if feature_names is not None:
            best_feature_name = feature_names[best_feature]

        # Creates inner nodes and root node of the tree
        node = Node(best_feature,
                    best_feature_name,
                    best_thresh,
                    left,
                    right,
                    best_gain,
                    gini=self._gini(y),
                    depth=depth,
                    value=leaf_value,
                    clf_value_dis=clf_value_dis,
                    clf_prob_dis=clf_prob_dis,
                    samples=n_samples)
        self.node_list.append(node)
        return node


    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for index in range(1,len(thresholds)):
                thr = (thresholds[index] + thresholds[index - 1]) / 2

                # calculate the information gain
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold, best_gain


    def _information_gain(self, y, X_column, threshold):

        # Information gain based on Gini or entropy
        criterion = self.criterion

        # parent entropy
        if criterion =="entropy":
            parent_entropy = self._entropy(y)
        elif criterion=="gini":
            parent_gini = self._gini(y)

        # create children
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # calculate the weighted avg. entropy/gini of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)

        if criterion =="entropy":
            e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
            child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

            # calculate the IG
            information_gain = parent_entropy - child_entropy

        elif criterion=="gini":
            g_l, g_r = self._gini(y[left_idxs]), self._gini(y[right_idxs])
            child_gini = (n_l/n) * g_l + (n_r/n) * g_r

            # calculate the IG (weighted impurity)
            information_gain = (n / self.no_samples_total) * (parent_gini -child_gini)

        return information_gain


    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p>0])

    def _gini(self, y):

        n = len(y)
        k = self.k

        if self.treetype=="classification":

            _, counts = np.unique(y, return_counts=True)

            probabilities = counts / counts.sum()  # Probability of each class

            impurity = 1 - sum(probabilities**2)

        elif self.treetype == "regression":
            if len(y) == 0:   # empty data
                impurity = 0
            else:
                impurity = np.mean((y-np.mean(y))**2) # RMSE

        # for binary case, finite sample correction, impurity is weighted by n/(n-1)
        if (k != None) and (n>k):
            impurity = impurity * n / (n-k)
        #elif (k != None) and (n<=k):
        #print("n<=k, error!")
        #print(n)
        return impurity

    def _mean_label(self,y):
        return np.mean(y)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _get_decision_paths(self):
        '''Create list of decision path from root node to each existing leaf'''
        self.decision_paths = list(self._paths(self.root)) # list of lists
        self.decision_paths_str = ["->".join(map(str, path)) for path in self.decision_paths] # string for readability


    def _paths(self, node, p = ()):
        '''Walks down each decision path from root node to each leaf and sets id for each node'''
        if node.left or node.right:
            if node.id is None:
                node.id = self.n_nodes
                self.n_nodes += 1
            yield from self._paths(node.left, (*p, node.id))
            yield from self._paths(node.right, (*p, node.id))
        else:
            if node.id is None:
                node.id = self.n_nodes
                self.n_nodes += 1
            yield (*p, node.id)

    def _apply_hierarchical_srinkage(self, treetype):

        if treetype=="regression":
            node_values_HS = np.zeros(len(self.node_list))

            #Iterate through all decision path
            for decision_path in self.decision_paths:

                #leaf_id = decision_path[-1]

                # Calculate telescoping sum as defined in orig. paper (https://proceedings.mlr.press/v162/agarwal22b/agarwal22b.pdf)
                cum_sum = 0
                for l, node_id in enumerate(decision_path):
                    if l==0:
                        cum_sum = self.node_list[node_id].value
                        node_values_HS[node_id] = cum_sum
                        continue

                    current_node = self.node_list[node_id]
                    node_id_parent = decision_path[l-1]
                    parent_node = self.node_list[node_id_parent]

                    cum_sum += ((current_node.value - parent_node.value) / (
                        1 + self.HS_lambda/parent_node.samples))

                    # Replace value of node with HS value outcome
                    node_values_HS[node_id] = cum_sum

            for node_id, value in enumerate(node_values_HS):
                self.node_list[node_id].value = value

        elif treetype=="classification":

            # Create deep copies of relevant attributes (otherwise original values will be overwritten)
            decision_paths = copy.deepcopy(self.decision_paths)
            clf_prob_dist = np.stack(copy.deepcopy([node_id.clf_prob_dis for node_id in self.node_list]), axis=0 )
            node_samples = copy.deepcopy([node_id.samples for node_id in self.node_list])

            #Get empty array with no rows == no of nodes and 2 columns (for class 0 and class 1)
            node_values_HS = np.zeros((len(node_samples),2))

            #Iterate through all decision path
            for decision_path in decision_paths:

                #Get empty array with no rows == no of nodes and 2 columns (for class 0 and class 1)
                node_values_ = np.zeros((len(node_samples),2))

                # Calculate telescoping sum as defined in orig. paper (https://proceedings.mlr.press/v162/agarwal22b/agarwal22b.pdf)
                cum_sum = 0
                for l, node_id in enumerate(decision_path):
                    if l == 0:
                        cum_sum = copy.deepcopy(clf_prob_dist[node_id])

                        node_values_[node_id] = cum_sum
                        continue

                    node_id_parent = decision_path[l - 1]

                    cum_sum += ((clf_prob_dist[node_id]-clf_prob_dist[node_id_parent])/(1 + self.HS_lambda / node_samples[node_id_parent]))

                    node_values_[node_id] = cum_sum
                for node_id in decision_path:
                    node_values_HS[node_id] = node_values_[node_id]

            # Update class attributes
            for node_id in range(len(self.node_list)):
                self.node_list[node_id].clf_prob_dis = node_values_HS[node_id]
                self.node_list[node_id].value = np.argmax(self.node_list[node_id].clf_prob_dis)

    def _get_feature_importance(self):

        feature_importance = np.zeros(self.n_features)
        features_list = [i.feature for i in self.node_list]
        feat_imp_p_node = np.nan_to_num(
            np.array([i.gain for i in self.node_list], dtype=float))

        for feat_num, feat_imp in zip(features_list, feat_imp_p_node):
            if feat_num is not None:
                feature_importance[feat_num] = feat_imp

        # Normalize information gain (total sum == 1)
        feature_importance_scaled = np.divide(feature_importance,
                                              (np.sum(feature_importance)))

        self.feature_importances_ = feature_importance_scaled

#class ClassificationTree(DecisionTree):
#
#    def __init__(self,
#                 min_samples_split=2,
#                 min_samples_leaf=1,
#                 max_depth=100,
#                 n_features=None,
#                 criterion="gini",
#                 treetype="classification",
#                 k=None,
#                 random_state=None):
#        super().__init__(min_samples_split, min_samples_leaf, max_depth,n_features,criterion, treetype, k, random_state)
