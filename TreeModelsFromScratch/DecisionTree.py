import numpy as np
import pandas as pd
from collections import Counter
import copy
import numbers
from warnings import warn, catch_warnings, simplefilter
#import ipdb

class Node:
    def __init__(self, feature=None, feature_name=None, threshold=None, left=None, right=None,
                 gain=None, id=None, depth=None, leaf_node=False, samples=None, gini=None,
                 value=None, clf_value_dis=None, clf_prob_dis=None):
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
    def __init__(self, min_samples_split=2, min_samples_leaf=1, max_depth=None, n_features=None, criterion="gini",
                 treetype="classification", k=None, feature_names=None, HShrinkage=False, HS_lambda=0, random_state=None):
        self.min_samples_split=min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth=max_depth
        self.n_features=n_features #for feature subsampling
        self.feature_names = feature_names #only relevant for Random Forests
        self.root=None
        self.criterion=criterion
        self.k=k
        self.treetype = treetype
        self.HShrinkage = HShrinkage
        self.HS_lambda = HS_lambda
        self.random_state = random_state
        self.random_state_ = self._check_random_state(random_state)
        #self.random_state = np.random.default_rng(random_state)
        #if isinstance(random_state, np.random._generator.Generator):
        #    self.random_state = random_state
        #else:
        #    self.random_state = np.random.default_rng(random_state)
        self.n_nodes=0
        self.oob_preds=None #only relevant for random forests oob
        self.oob_shap = None  #only relevant for random forests oob shap
        self.HS_applied = False # to store whether HS already was applied duing fit

    def _check_random_state(self, seed):
        if isinstance(seed, numbers.Integral) or (seed is None):
            return np.random.RandomState(seed)
            #return np.random.default_rng(seed)
        if isinstance(seed, np.random.RandomState):
            return seed

    def fit(self, X, y):
        #self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        if not self.n_features:
            self.n_features = X.shape[1]
        elif self.n_features=="sqrt":
            self.n_features = int(np.rint(np.sqrt(X.shape[1]))) #square root of number of feats in X
        else:
            self.n_features = min(X.shape[1], self.n_features)

        self.features_in_ = range(X.shape[1])
        self.node_list = []
        self.node_id_dict = {}
        self.no_samples_total = y.shape[0]

        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns
            #self.features_in_ = X.columns
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        self.root = self._grow_tree(X, y, feature_names=self.feature_names)

        # Set decision paths and node ids as class attributes
        self._get_decision_paths()

        #Sort node_list by node id
        self.node_list = sorted(self.node_list, key=lambda o: o.id)

        #Apply Hierarchical Shrinkage (change value (prediction) in leaf
        if self.HShrinkage:
            self._apply_hierarchical_srinkage(treetype=self.treetype)

        # Create dict of dict for all nodes with important attributes per node
        self._create_node_dict()

        #Set max tree depth as class attributes
        depth_list = [len(i) for i in self.decision_paths]
        self.max_depth_ = max(depth_list)-1

        #Set feature_importances_ (information_gain_scaled) as class attribute
        self._get_feature_importance(X)


    def _create_node_dict(self):
        '''Create dict of dict for all nodes with important attributes per node'''

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
                    node.id]["prob_distribution"] = list(node.clf_prob_dis)

    def _grow_tree(self, X, y, depth=0, feature_names=None):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        #Calculate node/leaf value
        if self.treetype == "classification":
            counter = Counter(y)
            clf_value_dis = [counter.get(0) or 0, counter.get(1) or 0]
            clf_prob_dis = (np.array(clf_value_dis) / n_samples)
            leaf_value = np.argmax(clf_prob_dis)

        elif self.treetype == "regression":
            leaf_value = self._mean_label(y)
            clf_value_dis = None
            clf_prob_dis = None

        # check the stopping criteria and creates leaf
        if ((self.max_depth is not None) and ((depth >= self.max_depth))
                or (n_labels == 1) or (n_samples < self.min_samples_split)
                or ((self.k!= None) and (n_samples <= self.k))):
            #or (n_samples < 2*self.min_samples_leaf)):
            node = self._create_leaf(leaf_value, clf_value_dis, clf_prob_dis,
                                y, depth, n_samples)
            return node

        # Random feature subsampling at each split point
        feat_idxs = self.random_state_.choice(n_feats, self.n_features, replace=False)

        # find the best split
        best_feature, best_thresh, best_gain = self._best_split(X, y, feat_idxs)

        # If no imporvement is found: Create leaf
        if (best_gain==-1) or (best_feature is None) or (best_thresh is None):
            node = self._create_leaf(leaf_value, clf_value_dis, clf_prob_dis,
                                y, depth, n_samples)
            return node

        # split samples in left and right child
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        #print(left_idxs, right_idxs)

        # If no of childs on one side is smaller than the parameter value: Create leaf
        if (len(left_idxs) < self.min_samples_leaf) or (
                len(right_idxs) < self.min_samples_leaf) or (
                    (self.k != None) and ((len(left_idxs) <= self.k) or
                                          (len(right_idxs) <= self.k))):
            node = self._create_leaf(leaf_value, clf_value_dis, clf_prob_dis,
                    y, depth, n_samples)
            return node

        # create child nodes
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


    def _create_leaf(self, leaf_value, clf_value_dis, clf_prob_dis, y, depth, n_samples):

        node = Node(value=leaf_value,
            clf_value_dis=clf_value_dis,
            clf_prob_dis = clf_prob_dis,
            leaf_node=True,
            gini=self._gini(y),
            depth=depth,
            samples=n_samples)
        self.node_list.append(node)

        return node



    def _best_split(self, X, y, feat_idxs):

        best_gain = np.array([-1])
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            # if only one threshold / value exist in column
            if len(thresholds)==1:
                gain = self._information_gain(y, X_column, thresholds[0])

                if gain > best_gain.max():
                    best_gain = np.array([gain])
                    split_idx = np.array([feat_idx])
                    split_threshold = thresholds

            for index in range(1,len(thresholds)):
                thr = (thresholds[index] + thresholds[index - 1]) / 2

                # calculate the information gain
                gain = self._information_gain(y, X_column, thr)

                #                if gain > best_gain:
                #                    best_gain = gain
                #                    split_idx = feat_idx
                #                    split_threshold = thr

                # if gain>best_gain --> replace gain, splitidx and thres


                if gain > best_gain.max():
                    best_gain = np.array([gain])
                    split_idx = np.array([feat_idx])
                    split_threshold = np.array([thr])


                # elif gain==best_gain --> create list and append
                elif gain == best_gain.all():
                    best_gain = np.append(best_gain, gain)
                    split_idx = np.append(split_idx, feat_idx)
                    split_threshold = np.append(split_threshold, thr)


        # Draw random gain/thres/feature from best splits
        idx_best = self.random_state_.choice(best_gain.shape[0], 1)[0]
        #print(best_gain)

        return split_idx[idx_best], split_threshold[idx_best], best_gain[idx_best]
        #return split_idx, split_threshold, best_gain


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
            return 0 #if no samples are left in either the left or right child return 0

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

            # calculate the IG (weighted impurity decrease) (rescaled gain) ==MDI
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
                impurity = np.mean((y-np.mean(y))**2) # MSE

        # for binary case, finite sample correction, impurity is weighted by n/(n-1)
        if (k != None) and (n>k):
            impurity = impurity * n / (n-k)
        elif (k != None) and (n<=k):
            impurity = 1
        #print("n<=k, error!")
        #print(n)
        return impurity

    def _mean_label(self,y):
        return np.mean(y)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        if isinstance(X, pd.Series):
            return np.array(self._traverse_tree(X, self.root))
        else:
            return np.array([self._traverse_tree(x, self.root) for x in X])

    def predict_proba(self, X):

        # If function is called on a regression tree return nothing
        if self.treetype != "classification":
            message = "This function is only available for classification tasks"
            warn(message)
            return

        if isinstance(X, pd.DataFrame):
            X = X.values

        if isinstance(X, pd.Series):
            return np.array(self._traverse_tree(X, self.root, pred_proba=True))
        else:
            return np.array([self._traverse_tree(x, self.root, pred_proba=True) for x in X])

    def _traverse_tree(self, x, node, pred_proba=False):
        if node.is_leaf_node():
            if pred_proba:
                return node.clf_prob_dis
            else:
                return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left, pred_proba)
        return self._traverse_tree(x, node.right, pred_proba)

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


    def traverse_explain_path(self, x, node=None, dict_list=None):
        """Return taken decision path in the tree for the given sample and dict with detailed information about decision path"""
        if dict_list is None:
            dict_list = []

        dict_node = {"node_id": node.id}

        if node.is_leaf_node():

            if self.treetype == "classification":
                dict_node.update([("value", node.value),
                                ("prob_distribution", node.clf_prob_dis)])
                dict_list.append(dict_node)
                return [dic.get("node_id") for dic in dict_list], dict_list
            dict_node["value"] = node.value
            dict_list.append(dict_node)
            return [dic.get("node_id") for dic in dict_list], dict_list

        dict_node.update([("feature", node.feature_name or node.feature),
                        ("threshold", np.round(node.threshold, 3)),
                        ("value_observation", x[node.feature].round(3))])

        if x[node.feature] <= node.threshold:
            dict_node["decision"] = "{} <= {} --> left".format(
                x[node.feature].round(3), np.round(node.threshold, 3))
            dict_list.append(dict_node)
            return self.traverse_explain_path(x, node.left, dict_list)
        dict_node["decision"] = "{} > {} --> right".format(
            x[node.feature].round(3), np.round(node.threshold, 3))
        dict_list.append(dict_node)
        return self.traverse_explain_path(x, node.right, dict_list)


    def explain_decision_path(self, X):
        """Takes one ore more observations and explains the samples path along the tree and returns decision path and detailed information about path"""
        if isinstance(X, pd.DataFrame):
            X = X.values

        if isinstance(X, pd.Series):
            return np.array(self.traverse_explain_path(X, self.root))

        return np.array([self.traverse_explain_path(x, self.root) for x in X])

    def _apply_hierarchical_srinkage(self, treetype=None, HS_lambda=None, smSHAP_coefs=None, m_nodes=None):

        if treetype==None:
            treetype = self.treetype
        if HS_lambda==None:
            HS_lambda = self.HS_lambda

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
                    
                    # Use HS with Smooth SHAP if coefs are given
                    if (smSHAP_coefs!=None):
                        cum_sum += ((current_node.value - parent_node.value) / (
                            1 + HS_lambda/parent_node.samples)) * np.abs(smSHAP_coefs[parent_node.feature])

                    # Use HS with nodewise smoothing if m_nodes are given
                    elif (m_nodes!=None):
                        cum_sum += ((current_node.value - parent_node.value) / (
                            1 + HS_lambda/parent_node.samples)) * m_nodes[node_id]

                    # Use Orignal HS
                    else:
                        cum_sum += ((current_node.value - parent_node.value) / (
                            1 + HS_lambda/parent_node.samples)) 

                    # Replace value of node with HS value outcome
                    node_values_HS[node_id] = cum_sum

            for node_id, value in enumerate(node_values_HS):
                self.node_list[node_id].value = value

        elif treetype=="classification":

            # Create deep copies of relevant attributes (otherwise original values will be overwritten)
            decision_paths = copy.deepcopy(self.decision_paths)
            #clf_prob_dist = np.stack(copy.deepcopy([node_id.clf_prob_dis for node_id in self.node_list]), axis=0 )
            clf_prob_dist = np.array(copy.deepcopy([node_id.clf_prob_dis for node_id in self.node_list]))
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

                    current_node = self.node_list[node_id]  
                    node_id_parent = decision_path[l - 1]
                    parent_node = self.node_list[node_id_parent]

                    # Use Selective HS using Smooth SHAP
                    if (smSHAP_coefs!=None):
                        cum_sum += ((clf_prob_dist[node_id]-clf_prob_dist[node_id_parent])/
                            (1 + HS_lambda / node_samples[node_id_parent])) * np.abs(smSHAP_coefs[parent_node.feature])

                    # Use HS with nodewise smoothing if m_nodes are given
                    elif (m_nodes!=None):
                        cum_sum += ((clf_prob_dist[node_id]-clf_prob_dist[node_id_parent])/
                            (1 + HS_lambda / node_samples[node_id_parent])) * m_nodes[node_id]

                    # Use Original HS
                    else:
                        cum_sum += ((clf_prob_dist[node_id]-clf_prob_dist[node_id_parent])/
                            (1 + HS_lambda / node_samples[node_id_parent]))

                    node_values_[node_id] = cum_sum
                for node_id in decision_path:
                    node_values_HS[node_id] = node_values_[node_id]

            # Update class attributes
            for node_id in range(len(self.node_list)):
                self.node_list[node_id].clf_prob_dis = node_values_HS[node_id]
                self.node_list[node_id].value = np.argmax(self.node_list[node_id].clf_prob_dis)

        #set attribute to indicate that HS was applied
        self.HS_applied = True

    def _get_feature_importance(self, X):

        feature_importance = np.zeros(X.shape[1]) # empty array in shape of n. features
        features_list = [i.feature for i in self.node_list] #feature per node (which was used to split)
        feat_imp_p_node = np.nan_to_num(np.array([i.gain for i in self.node_list], dtype=float)) ##gain per node (if nan then convert to 0)

        # for each node: add node.gain under feature_importance[feature_id]
        for feat_num, feat_imp in zip(features_list, feat_imp_p_node):
            if feat_num is not None:
                feature_importance[feat_num] += feat_imp

        # Normalize information gain (total sum == 1)
        if np.sum(feature_importance)!=0:
            feature_importance_scaled = np.divide(feature_importance,
                                              (np.sum(feature_importance)))
        # If no split was conducted do not scale because of division by 0 error
        else:
            feature_importance_scaled = feature_importance

        self.feature_importances_ = feature_importance_scaled


    def export_tree_for_SHAP(self, return_tree_dict=False):

        # Children
        children_left = []
        children_right = []

        # go over all nodes
        for node in self.node_list:

            # find child node id of corresponding node
            if node.left is not None:
                children_left.append(node.left.id)
            # if leaf return -1
            else:
                children_left.append(-1)

            # find child node id of corresponding node
            if node.right is not None:
                children_right.append(node.right.id)
            # if leaf return -1
            else:
                children_right.append(-1)

        children_left = np.array(children_left)
        children_right = np.array(children_right)
        children_default = children_right.copy()

        #features: replace "None" for features in leaf node with ""-2"
        features = np.array([
            node.feature if node.feature is not None else -2
            for node in self.node_list
        ])

        # Thresholds: replace "None" for thres in leaf node with ""-2"
        thresholds = np.array([
            node.threshold if node.threshold is not None else -2
            for node in self.node_list
        ])

        # values in array of arrays of shape (no_nodes, 1)
        if self.treetype == "regression":
            values = np.array([node.value for node in self.node_list])
        elif self.treetype == "classification":
            values = np.array([node.clf_prob_dis[1] for node in self.node_list])
        values = values.reshape(values.shape[0], 1)

        #samples
        samples = np.array([float(node.samples) for node in self.node_list])

        # define a custom tree model
        tree_dict = {
            "children_left": children_left,
            "children_right": children_right,
            "children_default": children_default,
            "features": features,
            "thresholds": thresholds,
            "values": values,
            "node_sample_weight": samples
        }
        model = {"trees": [tree_dict]}

        if return_tree_dict:
            return model, tree_dict
        return model

    def _get_parent_node(self, node_id):
        """Return parent node id for given node_id"""
        return [node.id for node in self.node_list if (node.leaf_node==False) if ((node.left.id==node_id)|(node.right.id==node_id))][0]

    def _reestimate_node_values(self, X, y):
        """Re-calculate value for each node based on given samples and returns list of new node values, dict with n_samples and node_val,
        and list of nan rows which were overwritten with new node values from parent node"""
        import ipdb; ipdb.set_trace()
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Get decision path (list of node ids) per observation
        traversed_nodes = self.explain_decision_path(X)[:, 0].copy()

        # Create array filled with nan to store y values for each node which is passed by the observation
        y_vals_array = np.full((self.n_nodes, X.shape[0]), np.nan) #shape: (n_nodes, n_observations)

        # Fill array
        for i, (idxs, y) in enumerate(zip(traversed_nodes, y)):
            y_vals_array[list(idxs),[i]] = y

        # get index of arrays which onl contain nan values (==node_id)
        nan_rows = np.argwhere(np.isnan(y_vals_array).all(axis=1)).flatten()

        # check if there are rows which only contain nan
        if nan_rows.shape[0] != 0:

            for nan_node_id in nan_rows:
                # get parent node for these rows
                par_node_id = self._get_parent_node(nan_node_id)

                # copy row from parent node and paste it into this row (so they will end up with same mean)
                y_vals_array[nan_node_id] = y_vals_array[par_node_id]

        # Dictonary to store results p node
        result = {}

        if self.treetype == "regression":

            node_vals = np.nanmean(y_vals_array, axis=1)
            n_samples = np.count_nonzero(~np.isnan(y_vals_array), axis=1)

            for i in range(y_vals_array.shape[0]):

                result[i]={"samples": n_samples[i],
                        "value": node_vals[i]
                    }

            #for regression also return list of node values p. node
            node_vals = np.nanmean(y_vals_array, axis=1)

            return node_vals, result, nan_rows, y_vals_array


        elif self.treetype == "classification":

            for i in range(y_vals_array.shape[0]):

                val, cnts = np.unique(y_vals_array[i,:][~np.isnan(y_vals_array[i,:])], return_counts=True)
                counts = {k: v for k, v in zip(val, cnts)}

                clf_value_dis = [counts.get(0) or 0, counts.get(1) or 0]
                n_samples = np.sum(clf_value_dis)

                clf_prob_dis = (np.array(clf_value_dis) / n_samples)
                leaf_value = np.argmax(clf_prob_dis)

                result[i]={"samples": n_samples,
                        "value": leaf_value,
                        "value_distribution": clf_value_dis,
                        "prob_distribution": clf_prob_dis
                    }

            #for classification also return list of value probabilities p. node
            node_prob = np.array([(1-val, val) for val in np.nanmean(y_vals_array, axis=1)])

            return node_prob, result, nan_rows, y_vals_array

    # def verify_shap_model(self, explainer, X):
    #     '''Verify the integrity of SHAP explainer model by comparing output of export_tree_for_SHAP vs original model'''

    #     if self.treetype=="classification":
    #         # Make sure that the ingested SHAP model (a TreeEnsemble object) makes the same predictions as the original model
    #         assert np.abs(
    #             explainer.model.predict(X) -
    #             self.predict_proba(X)[:, 1]).max() < 1e-4

    #         # make sure the SHAP values sum up to the model output (this is the local accuracy property)
    #         assert np.abs(explainer.expected_value +
    #                       explainer.shap_values(X).sum(1) -
    #                       self.predict_proba(X)[:, 1]).max() < 1e-4
    #     else:
    #         # Make sure that the ingested SHAP model (a TreeEnsemble object) makes the same predictions as the original model
    #         assert np.abs(explainer.model.predict(X) -
    #                       self.predict(X)).max() < 1e-4

    #         # make sure the SHAP values sum up to the model output (this is the local accuracy property)
    #         assert np.abs(explainer.expected_value +
    #                       explainer.shap_values(X).sum(1) -
    #                       self.predict(X)).max() < 1e-4

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
