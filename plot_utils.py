import sklearn
import seaborn as sns; sns.set_style("ticks")
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve, CalibrationDisplay
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import re 

def plot_multiclass_pr_from_preds(y_test, y_pred, ax, 
                        title="PR Curve", labels=None, figsize=(8, 6)):

    import pandas as pd
    # structures
    precision = dict()
    recall = dict()
    
    n_classes = y_pred.shape[1]
    assert y_pred.shape[1] >= np.max(y_test)

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False)
    y_test_dummies_vals = y_test_dummies.values
    
    average_precision = average_precision_score(y_test_dummies_vals, y_pred, average=None)
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_dummies_vals[:, i], y_pred[:, i])
        # print(precision[i][-50:], recall[i][-50:])
        
    # pr for each class
    # fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0.5, 0.5], 'k--')
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.5, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)

    if labels:
        for i, label in zip(range(n_classes), labels):
            ax.plot(precision[i], recall[i], label=f'PR curve (AP = {average_precision[i]:.2f}) for {label}')
    else:
        for i in range(n_classes):
            ax.plot(precision[i], recall[i], label=f'PR curve (AP = {average_precision[i]:.2f}) for label {i}')
        
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    # plt.show()

def plot_multiclass_roc_from_preds(y_test, y_pred, ax, 
                        title="ROC Curve", labels=None, figsize=(8, 6)):
    import pandas as pd
    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    n_classes = y_pred.shape[1]
    # assert y_pred.shape[1] >= np.max(y_test)
    
    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False)
    y_test_dummies_vals = y_test_dummies.values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies_vals[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # roc for each class
    # fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    if labels:
        for i, label in zip(range(n_classes), labels):
            ax.plot(fpr[i], tpr[i], label=f'ROC curve (area = {roc_auc[i]:.2f}) for {label}')
    else:
        for i in range(n_classes):
            ax.plot(fpr[i], tpr[i], label=f'ROC curve (area = {roc_auc[i]:.2f}) for label {i}')
        
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    # sns.despine()
    # plt.show()
