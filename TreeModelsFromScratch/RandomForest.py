from TreeModelsFromScratch.DecisionTree import DecisionTree
import numpy as np
import pandas as pd
#from collections import Counter
from warnings import warn, catch_warnings, simplefilter
from sklearn.metrics import mean_squared_error, accuracy_score
import numbers
from shap.explainers._tree import SingleTree
from shap import TreeExplainer
from TreeModelsFromScratch.SmoothShap import verify_shap_model, smooth_shap, conf_int_ratio_two_var, conf_int_cohens_d, conf_int_ratio_mse_ratio

class RandomForest:
    def __init__(self,
                 n_trees=10,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 n_feature="sqrt",
                 bootstrap=True,
                 oob=True,
                 oob_SHAP=False,
                 criterion="gini",
                 treetype="classification",
                 HShrinkage=False,
                 HS_lambda=0,
                 HS_smSHAP=False,
                 HS_nodewise_shrink_type=None,
                 cohen_reg_param=2,
                 alpha=0.05,
                 cohen_statistic="f",
                 k=None,
                 random_state=None,
                 testHS=False):
        self.n_trees = n_trees
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.min_samples_leaf = min_samples_leaf  # Still need to be implemented
        self.n_features=n_feature
        self.bootstrap=bootstrap
        self.oob = oob
        self.oob_SHAP=oob_SHAP #for calculation of shap scores for oob predictions
        self.criterion = criterion
        self.k = k
        self.HShrinkage = HShrinkage
        self.HS_lambda = HS_lambda
        self.HS_smSHAP = HS_smSHAP # for smooth SHAP hierarchical shrinkage
        self.HS_nodewise_shrink_type = HS_nodewise_shrink_type #For nodewise smoothing ("MSE_ratio" or "effect_size")
        self.cohen_reg_param = cohen_reg_param #For nodewise smoothing
        self.alpha = alpha #For nodewise smoothing
        self.cohen_statistic = cohen_statistic #For nodewise smoothing
        self.treetype = treetype
        self.random_state = random_state
        self.random_state_ = self._check_random_state(random_state)
        #self.random_state = np.random.default_rng(random_state)
        self.trees = []
        self.feature_names = None
        self.smSHAP_HS_applied = False
        self.nodewise_HS_applied = False
        self.testHS = testHS

    def _check_random_state(self, seed):
        if isinstance(seed, numbers.Integral) or seed==None:
            return np.random.RandomState(seed)
            #return np.random.default_rng(seed)
        if isinstance(seed, np.random.RandomState):
            return seed

    def fit(self, X, y):
        self.trees = []

        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        if self.oob:
            #empty list of lists to keep track of which tree predicted each oob observation (only for analyzing/debugging purposes)
            self.oob_preds_tree_id = [
                [] for _ in range(X.shape[0])
            ]

        if self.oob_SHAP:
            #Create array filled with nans in shape [n_obs, n_feats, n_trees] for shap oob
            shap_scores_inbag = np.full([X.shape[0], X.shape[1], self.n_trees], np.nan)
            shap_scores_oob = np.full([X.shape[0], X.shape[1], self.n_trees], np.nan)

        #Empty array to store individual feature importances p. tree in the forest
        feature_importance_trees = np.empty((self.n_trees, X.shape[1]))

        #Create random seeds for each tree in the forest
        MAX_INT = np.iinfo(np.int32).max
        seed_list = self.random_state_.randint(MAX_INT, size=self.n_trees)

        #Create forest
        for i, seed in zip(range(self.n_trees), seed_list):
            #for _ in range(self.n_trees):

            #Instantiate tree
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                min_samples_leaf=self.min_samples_leaf,
                                n_features=self.n_features,
                                criterion=self.criterion,
                                treetype=self.treetype,
                                feature_names=self.feature_names,
                                HShrinkage=self.HShrinkage,
                                HS_lambda=self.HS_lambda,
                                k=self.k,
                                random_state=seed)#self.random_state)

            #Draw bootstrap samples (inbag)
            X_inbag, y_inbag, idxs_inbag = self._bootstrap_samples(
                X, y, self.bootstrap, self.random_state_) #self._check_random_state(seed))

            # Fit tree using inbag samples
            tree.fit(X_inbag, y_inbag)
            self.trees.append(tree) #Add tree to forest
            feature_importance_trees[i, :] = tree.feature_importances_ #add feature importance to array

            # Draw oob samples (which have not been used for training) and predict oob observations
            if self.oob:
                n_samples = X.shape[0]
                tree.oob_preds = np.full(n_samples, np.nan)#np.zeros(n_samples, dtype=np.float64)
                #n_oob_pred = np.zeros(n_samples, dtype=np.int64)

                X_oob, y_oob, idxs_oob = self._oob_samples(X, y, idxs_inbag)

                tree.oob_preds[idxs_oob] = tree.predict(X_oob)

                for j in idxs_oob:
                    self.oob_preds_tree_id[j].append(i)

                # Apply nodewise HS
                if self.HS_nodewise_shrink_type != None:
                    self.apply_nodewise_HS(tree, X_inbag, y_inbag, X_oob, y_oob, shrinkage_type=self.HS_nodewise_shrink_type, HS_lambda=self.HS_lambda, cohen_reg_param=self.cohen_reg_param, alpha=self.alpha, cohen_statistic=self.cohen_statistic, testHS=self.testHS)

                # Compute inbag and oob SHAP values
                if self.oob_SHAP:

                    #Create array with nan for single tree shap values which can be pasted in shap_scores_oob array
                    shap_scores_inbag_tree = np.full([X.shape[0], X.shape[1]], np.nan)
                    shap_scores_oob_tree = np.full([X.shape[0], X.shape[1]], np.nan)

                    #Create shap explainer for individual tree
                    export_tree = tree.export_tree_for_SHAP()
                    explainer_tree = TreeExplainer(export_tree)
                    verify_shap_model(tree, explainer_tree, X_inbag)

                    #Calculate shap scores for oob
                    shap_tree_inbag = explainer_tree.shap_values(X_inbag)
                    shap_tree_oob = explainer_tree.shap_values(X_oob)

                    #Put shap oob scores in correct position of array (correct idx of observation)
                    np.put_along_axis(shap_scores_inbag_tree,
                                      idxs_inbag.reshape(idxs_inbag.shape[0], 1),
                                      shap_tree_inbag,
                                      axis=0)
                    np.put_along_axis(shap_scores_oob_tree,
                                      idxs_oob.reshape(idxs_oob.shape[0], 1),
                                      shap_tree_oob,
                                      axis=0)

                    # Update values of overall shap_scores_oob array
                    shap_scores_inbag[:, :, i] = shap_scores_inbag_tree.copy()
                    shap_scores_oob[:, :, i] = shap_scores_oob_tree.copy()

        # Calculate and set feature importance of forest as class attribute
        self.feature_importances_ = feature_importance_trees.mean(axis=0)

        # Calculate oob_score for all trees within forest
        if self.oob:

            #surpress unnecessary np.nanmean error
            with catch_warnings():
                simplefilter("ignore", category=RuntimeWarning)

                # Get mean value for each oob prediction ignoring the nan values (nan will be kept only if there is no prediction from none of the trees in the forest)
                self.oob_preds_forest = np.nanmean([tree.oob_preds for tree in self.trees], axis=0)
            y_test_oob = y.copy()

            # Check if there are any y obs where there is no oob prediction:
            if np.isnan(self.oob_preds_forest).any():

                # identify index of all nan values (where no oob pred is found)
                nan_indxs = np.argwhere(np.isnan(self.oob_preds_forest))

                #Throw UserWarning of how many values did not have an oob prediction
                message = """{} out of {} samples do not have OOB scores. This probably means too few trees were used to compute any reliable OOB estimates. These samples were dropped before computing the oob_score""".format(len(nan_indxs), len(y))
                warn(message)

                # drop these NaN values from X_oob_preds and y_test_oob
                mask = np.ones(self.oob_preds_forest.shape[0], dtype=bool)
                mask[nan_indxs] = False
                self.oob_preds_forest = self.oob_preds_forest[mask]
                y_test_oob = y[mask]

            # calculate oob_score and store score as class attribute
            if self.treetype=="classification":
                self.oob_preds_forest = self.oob_preds_forest
                self.oob_score = accuracy_score(
                    y_test_oob, self.oob_preds_forest.round(0))  #round to full number 0 or 1 for accuracy
            elif self.treetype=="regression":
                self.oob_score = mean_squared_error(y_test_oob, self.oob_preds_forest, squared=False) #RMSE

            #set attribute to store that nodewise HS was used
            if self.HS_nodewise_shrink_type != None:
                self.nodewise_HS_applied = True

            # Calculate average shap scores inbag and oob
            if self.oob_SHAP:
                self.inbag_SHAP_values = np.nanmean(shap_scores_inbag, axis=2)
                self.oob_SHAP_values = np.nanmean(shap_scores_oob, axis=2)

                # Apply Smooth SHAP HS
                if self.HS_smSHAP:
                    self.apply_smSHAP_HS(HS_lambda=self.HS_lambda)


    def _bootstrap_samples(self, X, y, bootstrap, random_state):

        if bootstrap:
            n_samples = X.shape[0]
            idxs_inbag = random_state.choice(n_samples, n_samples, replace=True)
            return X[idxs_inbag], y[idxs_inbag], idxs_inbag
        else:
            return X, y, np.arange(X.shape[0])

    def _oob_samples(self, X, y, idxs_inbag):
        mask = np.ones(X.shape[0], dtype=bool)
        mask[idxs_inbag] = False
        X_oob = X[mask]
        y_oob = y[mask]
        idxs_oob = mask.nonzero()[0]
        return X_oob, y_oob, idxs_oob

    def predict_proba(self, X):

        # If function is called on a regression tree return nothing
        if self.treetype != "classification":
            message = "This function is only available for classification tasks. This model is of type {}".format(
                self.treetype)
            warn(message)
            return

        if isinstance(X, pd.DataFrame):
            X = X.values

        predictions = np.array([tree.predict_proba(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)

        predictions = np.array([np.mean(pred, axis=0) for pred in tree_preds])

        return predictions

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.treetype=="regression":
            predictions = np.array([tree.predict(X) for tree in self.trees])
            tree_preds = np.swapaxes(predictions, 0, 1)
            predictions = np.mean(tree_preds, axis=1)
            return predictions

        elif self.treetype=="classification":
            predictions = np.argmax(self.predict_proba(X),axis=1)
            return predictions

    def export_forest_for_SHAP(self):
        tree_dicts = []
        for tree in self.trees:

            _, tree_dict = tree.export_tree_for_SHAP(return_tree_dict=True)

            tree_dicts.append(tree_dict)

        if self.treetype=="regression":
            # model = {
            #     "trees":[SingleTree(t, scaling=1.0 / len(tree_dicts)) for t in tree_dicts],
            #     #"base_offset": scipy.special.logit(orig_model2.init_.class_prior_[1]),
            #     "tree_output": "raw_value",
            #     "scaling": 1.0 / len(tree_dicts),
            #     "objective": "squared_error",
            #     "input_dtype": np.
            #     float32,  # this is what type the model uses the input feature data
            #     "internal_dtype": np.
            #     float64  # this is what type the model uses for values and thresholds
            # }
            model = [
                SingleTree(t, scaling=1.0 / len(tree_dicts))
                for t in tree_dicts
            ]
        elif self.treetype=="classification":
            # model = {
            #     #"trees": tree_dicts,
            #     "trees":[SingleTree(t, scaling=1.0 / len(tree_dicts)) for t in tree_dicts],
            #     #"base_offset":0.6274165202108963,  #scipy.special.logit(orig_model2.init_.class_prior_[1]),
            #     "tree_output": "probability",
            #     "scaling": 1.0/len(tree_dicts),
            #     "objective": "binary_crossentropy",
            #     "input_dtype": np.float32,  # this is what type the model uses the input feature data
            #     "internal_dtype": np.float64  # this is what type the model uses for values and thresholds
            # }
            model = [
                SingleTree(t, scaling=1.0 / len(tree_dicts), normalize=True)
                for t in tree_dicts
            ]
        return model

    def apply_smSHAP_HS(self, HS_lambda=0):
        '''Apply Selective HS using Smooth SHAP. Overwrites values of fitted tree. Can also be applied post hoc'''

        #check if forest already used HS during training: if yes, return error
        if (self.trees[0].HS_applied==True) | (self.smSHAP_HS_applied==True):
            message = "For the given model (selective) hierarchical shrinkage was already applied during fit! Please use an estimator with HSShrinkage=False & HS_smSHAP=False"
            warn(message)
            return

        # Calculate Smooth SHAP scores
        smSHAP_vals, _, smSHAP_coefs = smooth_shap(self.inbag_SHAP_values, self.oob_SHAP_values)
        self.smSHAP_coefs = smSHAP_coefs
        self.smSHAP_vals = smSHAP_vals

        #For each tree in the forest apply HS with sm SHAP lin coef
        for tree in self.trees:

            tree.HS_lambda = HS_lambda #update attribute HS_lambda
            tree._apply_hierarchical_srinkage(HS_lambda=HS_lambda, smSHAP_coefs=smSHAP_coefs) #apply HS with SmSHAP
            tree._create_node_dict() # Update node dict attributes for each tree

        #set attribute to store that smSHAP HS wasused
        self.smSHAP_HS_applied=True

    def apply_nodewise_HS(self, tree, X_inbag, y_inbag, X_oob, y_oob, shrinkage_type="MSE_ratio", HS_lambda=0, cohen_reg_param=2, alpha=0.05, cohen_statistic="f", testHS=False):
        '''Apply HS using smoothing coefficient based on discrepancies between inbag and oob data. Overwrites values of fitted tree. Can also be applied post hoc'''

        #check if forest already used HS during training: if yes, return error
        if (tree.HS_applied==True) | (self.nodewise_HS_applied==True):
            message = "For the given model (selective) hierarchical shrinkage was already applied during fit! Please use an estimator with HSShrinkage=False & HS_nodewise=False"
            warn(message)
            return

        # Reestimate node values for inbag/oob smoothing
        _, reest_node_vals_inbag, nan_rows_inbag, y_inbag_p_node = tree._reestimate_node_values(X_inbag, y_inbag)
        _, reest_node_vals_oob, nan_rows_oob, y_oob_p_node = tree._reestimate_node_values(X_oob, y_oob)

        # Variables to store results p node
        conf_int_nodes = []
        m_nodes = []

        # For each node calculate shrinkage param
        for i in range(tree.n_nodes):

            # Pass y_vals_inbag and oob to one of the conf int function
            if shrinkage_type=="MSE_ratio":
                conf_int, m = conf_int_ratio_mse_ratio(y_inbag_p_node[i,:][~np.isnan(y_inbag_p_node[i,:])], #filter out nans
                                                        y_oob_p_node[i,:][~np.isnan(y_oob_p_node[i,:])], #filter out nans
                                                        tree.node_list[i].value,
                                                        node_dict_inbag = reest_node_vals_inbag[i],
                                                        node_dict_oob = reest_node_vals_oob[i],
                                                        alpha=alpha, type=tree.treetype)
                conf_int_nodes.append(conf_int)
                m_nodes.append(m)
            elif shrinkage_type=="effect_size":
                conf_int, m = conf_int_cohens_d(y_inbag_p_node[i,:][~np.isnan(y_inbag_p_node[i,:])], #filter out nans
                                                    y_oob_p_node[i,:][~np.isnan(y_oob_p_node[i,:])], #filter out nans
                                                    reg_param=cohen_reg_param, alpha=alpha, cohen_statistic=cohen_statistic)
                conf_int_nodes.append(conf_int)
                m_nodes.append(m)

        # apply HS with smoothing m parameter
        tree._apply_hierarchical_srinkage(HS_lambda=HS_lambda, m_nodes=m_nodes, testHS=testHS) #apply HS with nodewise HS
        tree._create_node_dict() # Update node dict attributes for each tree

        # store m_nodes, conf_interval_nodes and other parameter settings as class attribute
        tree.nodewise_HS_dict = {"conf_intervals": conf_int_nodes,
                                "m_values": m_nodes,
                                "shrinkage_type":shrinkage_type,
                                "alpha":alpha,
                                "reest_node_vals_inbag":reest_node_vals_inbag,
                                "nan_rows_inbag":nan_rows_inbag,
                                "reest_node_vals_oob":reest_node_vals_oob,
                                "nan_rows_oob":nan_rows_oob}

        # Add additional information for shrinkage type effect size to dict
        if shrinkage_type=="effect_size":
            tree.nodewise_HS_dict["cohen_reg_param"]=cohen_reg_param
            tree.nodewise_HS_dict["cohen_statistic"]=cohen_statistic
