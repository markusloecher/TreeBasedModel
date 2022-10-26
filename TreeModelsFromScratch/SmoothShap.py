import numpy as np
import pandas as pd
#import shap
from sklearn.linear_model import LinearRegression
from sklearn.base import clone
from warnings import warn, catch_warnings, simplefilter
from copy import deepcopy
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


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
        smooth_shap_vals.append(linear_reg.predict(shap_values_inbag[:,i].reshape(-1, 1)))

    # Turn into array of shape (rows = observations, col=feats)
    smooth_shap_vals = np.array(smooth_shap_vals).T

    # Mean smooth shap value per feature
    mean_smooth_shap = np.mean(smooth_shap_vals, axis=0)

    if detailed_output:
        return smooth_shap_vals, mean_smooth_shap, lin_models, shap_values_inbag, shap_values_oob
    return smooth_shap_vals, mean_smooth_shap

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
        estimator_copy.fit(X_train, y_train)
        y_pred = estimator_copy.predict(X_test)
        scores.append(scoring_func(y_test, y_pred))

    return scores
