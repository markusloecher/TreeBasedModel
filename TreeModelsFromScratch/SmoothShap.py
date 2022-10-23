import numpy as np
import pandas as pd
#import shap
from sklearn.linear_model import LinearRegression
from sklearn.base import clone
from warnings import warn, catch_warnings, simplefilter

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
    preds = []

    # Check if there are any y obs where there is no oob prediction:
    if np.isnan(shap_values_oob).any():

        # identify index of all nan values (where no oob pred is found)
        nan_indxs = np.argwhere(np.isnan(shap_values_oob))

        #Throw UserWarning of how many values did not have an oob prediction
        message = """{} out of {} samples do not have OOB shap values. This probably means too few trees were used to compute any reliable OOB estimates. These samples were dropped before computing the smooth shap score""".format(
            int(nan_indxs.shape[0] / shap_values_oob.shape[1]), len(shap_values_oob))
        warn(message)

        # drop these NaN values from X_oob_preds and y_test_oob
        mask = np.ones(shap_values_oob.shape[0], dtype=bool)
        mask[nan_indxs] = False
        shap_values_inbag = shap_values_oob[mask]
        shap_values_oob = shap_values_oob[mask]



    #for each feature fit linear regression shap_oob = shap_inbag*beta+u1 and predict shapp oob
    for i in range(shap_values_inbag.shape[1]):
        linear_reg = LinearRegression().fit(shap_values_inbag[:, i].reshape(-1, 1), shap_values_oob[:, i])
        lin_models.append(clone(linear_reg))
        preds.append(linear_reg.predict(shap_values_inbag[:,i].reshape(-1, 1)))

    # Mean smooth shap value per feature
    smooth_shap = np.mean(preds, axis=1)

    if detailed_output:
        return smooth_shap, preds, lin_models, shap_values_inbag, shap_values_oob
    return smooth_shap
