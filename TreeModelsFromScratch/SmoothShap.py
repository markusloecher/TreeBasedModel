import numpy as np
import pandas as pd
#import shap
from sklearn.linear_model import LinearRegression
#from sklearn.base import clone
from warnings import warn#, catch_warnings, simplefilter
from copy import deepcopy
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from shap.explainers._tree import SingleTree
import itertools


# Utility functions to verify SHAP model which has been created based on tree output
def verify_shap_model(tree, explainer, X):
    '''Verify the integrity of SHAP explainer model by comparing output of export_tree_for_SHAP vs original model'''

    if tree.treetype=="classification":
        # Make sure that the ingested SHAP model (a TreeEnsemble object) makes the same predictions as the original model
        assert np.abs(
            explainer.model.predict(X) -
            tree.predict_proba(X)[:, 1]).max() < 1e-4

        # make sure the SHAP values sum up to the model output (this is the local accuracy property)
        assert np.abs(explainer.expected_value +
                    explainer.shap_values(X).sum(1) -
                    tree.predict_proba(X)[:, 1]).max() < 1e-4
    else:
        # Make sure that the ingested SHAP model (a TreeEnsemble object) makes the same predictions as the original model
        assert np.abs(explainer.model.predict(X) -
                    tree.predict(X)).max() < 1e-4

        # make sure the SHAP values sum up to the model output (this is the local accuracy property)
        assert np.abs(explainer.expected_value +
                    explainer.shap_values(X).sum(1) -
                    tree.predict(X)).max() < 1e-4


def smooth_shap(shap_values_inbag, shap_values_oob, detailed_output=False):
    """Compute smooth shap score per feature based on oob and inbag correlations"""

    lin_models = []
    lin_coefs = []
    smooth_shap_vals = []

    # Check if there are any y obs where there is no oob prediction:
    if np.isnan(shap_values_oob).any():

        # identify index of all nan values (where no oob or inbag pred is found)
        nan_indxs = np.argwhere((np.isnan(shap_values_inbag))
                                | (np.isnan(shap_values_oob)))

        #Throw UserWarning of how many values did not have an oob prediction
        message = """{} out of {} samples do not have Inbag or OOB shap values. This probably means too few trees were used to compute any reliable OOB estimates. These samples were dropped before computing the smooth shap score""".format(
            int(nan_indxs.shape[0] / shap_values_oob.shape[1]), len(shap_values_oob))
        warn(message)

        # drop these NaN values from x (shap inbag) and y (shap oob)
        mask = np.ones(shap_values_oob.shape[0], dtype=bool)
        mask[nan_indxs[:, 0]] = False
        shap_values_inbag = shap_values_inbag[mask]
        shap_values_oob = shap_values_oob[mask]

    #for each feature fit linear regression shap_oob = shap_inbag*beta+u1 and predict shap oob
    for i in range(shap_values_inbag.shape[1]):
        linear_reg = LinearRegression().fit(shap_values_inbag[:, i].reshape(-1, 1), shap_values_oob[:, i])
        lin_models.append(linear_reg)
        lin_coefs.append(linear_reg.coef_[0])
        smooth_shap_vals.append(linear_reg.predict(shap_values_inbag[:,i].reshape(-1, 1)))

    # Turn into array of shape (rows = observations, col=feats)
    smooth_shap_vals = np.array(smooth_shap_vals).T

    # Mean absolte smooth shap value per feature
    mean_smooth_shap = np.abs(smooth_shap_vals).mean(axis=0)

    if detailed_output:
        return smooth_shap_vals, mean_smooth_shap, lin_coefs, lin_models, shap_values_inbag, shap_values_oob
    
    return smooth_shap_vals, mean_smooth_shap, lin_coefs

def cross_val_score_scratch(estimator, X, y, cv=10, scoring_func=roc_auc_score, shuffle=True, random_state=None):
    '''Perform k-fold cross validation scoring for estimators with .fit and .predict function (imodels and scratch models)'''

    kf = KFold(n_splits=cv, shuffle=shuffle, random_state=random_state)
    scores = []

    for train_index, test_index in kf.split(X):

        # Create true copy of estimator (refitting of scratch models is not possbile)
        estimator_copy = deepcopy(estimator)

        #split data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #fit estimator, predict & score
        estimator_copy.fit(X=X_train, y=y_train)
        y_pred = estimator_copy.predict(X_test)
        scores.append(scoring_func(y_test, y_pred))

    return scores

def GridSearchCV_scratch(estimator, grid, X, y, cv=10, scoring_func=None, fit_best_est=True, shuffle=True, random_state=None):

    valid_grid_keys = [
        "reg_param", "n_trees", "HS_lambda", "max_depth", "min_samples_split",
        "min_samples_leaf", "k", "n_features", "n_feature", "HShrinkage"
    ]

    # Check if keys are valid
    for key in grid.keys():
        if key not in valid_grid_keys:
            message = """{} is not a valid hyperparameter for this function. Key has to be part of the following list
            {}""".format(key, valid_grid_keys)
            warn(message)
            break

    keys = list(grid.keys())

    #Get all possible combinations of hyperparameters from grid
    pos_combs = list(itertools.product(*grid.values()))

    cv_scores = np.zeros((len(pos_combs), cv)) #array to store cv results

    #iterrate over all possible combinations
    for j, param_comb in enumerate(pos_combs):

        # Create deep copy of estimator (necessary for scratch models)
        estimator_copy = deepcopy(estimator)

        #set attributes in param_comb
        for i, key in enumerate(keys):
            setattr(estimator_copy, key, param_comb[i])

        # k fold cross validation
        cv_scores[j,:] = cross_val_score_scratch(estimator_copy, X, y, cv=cv, scoring_func=scoring_func,
                                            shuffle=shuffle, random_state=random_state)


    # find best combination (highest avg. score accross all folds)
    cv_scores_mean = cv_scores.mean(axis=1)
    idx_best_comb = cv_scores_mean.argmax()

    results = {
        "best_param_comb": pos_combs[idx_best_comb],
        "best_test_score": cv_scores_mean[idx_best_comb],
        "mean_test_scores": cv_scores_mean,
        "param_combinations": pos_combs,
        "cv_scores_p_split": cv_scores
    }

    if fit_best_est is False:
        return results

    #fit estimator with best found combination
    for i, key in enumerate(keys):
        setattr(estimator, key, pos_combs[idx_best_comb][i])

    estimator.fit(X, y) #original estimaotr will be fitted inplace does not need to be returned

    return results


def export_imodels_for_SHAP(imodel, is_forest=True):
    '''Exports imodel estimator in a shap readable format'''
    if is_forest:

        tree_dicts = []

        for tree in imodel.estimator_.estimators_:

            # extract the arrays that define the tree
            children_left = tree.tree_.children_left
            children_right = tree.tree_.children_right
            children_default = children_right.copy(
            )  # because sklearn does not use missing values
            features = tree.tree_.feature
            thresholds = tree.tree_.threshold
            if str(type(imodel))[-12:-2] == "Classifier":
                values = tree.tree_.value[:, :, 1]
            else:
                values = tree.tree_.value.reshape(tree.tree_.value.shape[0], 1)
            node_sample_weight = tree.tree_.weighted_n_node_samples

            # define a custom tree model
            tree_dict = {
                "children_left": children_left,
                "children_right": children_right,
                "children_default": children_default,
                "features": features,
                "thresholds": thresholds,
                "values": values,
                "node_sample_weight": node_sample_weight
            }

            tree_dicts.append(tree_dict)

        if str(type(imodel))[-12:-2] == "Classifier":
            model = [
                SingleTree(t, scaling=1.0 / len(tree_dicts), normalize=True)
                for t in tree_dicts
            ]
        else:
            model = [
                SingleTree(t, scaling=1.0 / len(tree_dicts))
                for t in tree_dicts
            ]

        return model

    # If single Tree model

    # extract the arrays that define the tree
    children_left = imodel.estimator_.tree_.children_left
    children_right = imodel.estimator_.tree_.children_right
    children_default = children_right.copy(
    )  # because sklearn does not use missing values
    features = imodel.estimator_.tree_.feature
    thresholds = imodel.estimator_.tree_.threshold

    if str(type(imodel))[-12:-2] == "Classifier":
        values = imodel.estimator_.tree_.value[:, :, 1]
    else:
        values = imodel.estimator_.tree_.value.reshape(
            imodel.estimator_.tree_.value.shape[0], 1)
    node_sample_weight = imodel.estimator_.tree_.weighted_n_node_samples

    # define a custom tree model
    tree_dict = {
        "children_left": children_left,
        "children_right": children_right,
        "children_default": children_default,
        "features": features,
        "thresholds": thresholds,
        "values": values,
        "node_sample_weight": node_sample_weight
    }
    model = {"trees": [tree_dict]}

    return model
