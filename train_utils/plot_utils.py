import sklearn
import seaborn as sns; sns.set_style("ticks")
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve, CalibrationDisplay
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import re 

def plot_multiclass_pr(precision, recall, average_precision,
                        title="PR Curve", labels=None, figsize=(8, 6), n_classes=4, fname="AUPRC.png"):

    # structures
    # precision = dict()
    # recall = dict()
        
    # pr for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0.5, 0.5], 'k--')
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.5, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)

    if labels:
        for i in range(n_classes):
            ax.plot(precision[i], recall[i], label=f'PR curve (AP = {average_precision[i]:.2f}) for {labels[i]}')
    else:
        for i in range(n_classes):
            ax.plot(precision[i], recall[i], label=f'PR curve (AP = {average_precision[i]:.2f}) for label {i}')
        
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    # plt.show()
    fig.savefig(fname, dpi=300)    
    print("saved figure pr")
    

def plot_multiclass_roc(fpr, tpr, roc_auc,
                        title="ROC Curve", labels=None, figsize=(8, 6), n_classes=4, fname="AUROC.png"):

    # structures
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    
    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    if labels:
        for i in range(n_classes):
            ax.plot(fpr[i], tpr[i], label=f'ROC curve (area = {roc_auc[i]:.2f}) for {labels[i]}')
    else:
        for i in range(n_classes):
            ax.plot(fpr[i], tpr[i], label=f'ROC curve (area = {roc_auc[i]:.2f}) for label {i}')
        
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    fig.savefig(fname, dpi=300)    
    print("saved figure roc")
    # plt.savefig(fname)

def plot_roc(y_test, y_pred,
                    title="ROC Curve", label=None, figsize=(17, 6)):

    fpr, tpr, _ = roc_curve(y_test, y_pred[:, 1])
    print(fpr, tpr)
    roc_auc = auc(fpr, tpr)

    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    if label:
        ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for {label}')
    else:
        ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for label')
        
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    plt.show()
    
def plot_missclassified_hists(df_correct, df_misclass, cols,count_threshold=20, logy=True, figsize=(8,15),
                              title="Histograms", title_adjustment=0.93):
    
    ncols = 2
    fig, axs = plt.subplots(nrows=int(np.round(len(cols)/2)), ncols=ncols, figsize=figsize)
    fig.suptitle(title, fontsize=18)
    
    for col, ax in zip(cols, axs.ravel()):
        is_counts = True
        try:
            df_correct[col].astype(pd.Int64Dtype()) 
            df_misclass[col].astype(pd.Int64Dtype()) 
            # if there are more than a certain number of values, it's probably not a count
            if len(df_correct[col].value_counts()) > count_threshold:
                is_counts = False
#             ignore age, since they're all integers but it's not a count variable                
            if col == "age":
                is_counts = False
        except:
            is_counts = False
            print(col)
        
        if df_correct[col].dtype == "object" or is_counts:
            corr_value_counts = df_correct[col].value_counts().to_frame().rename({col:"Correct"}, axis=1)/df_correct[col].value_counts().sum()
            wrong_value_counts = df_misclass[col].value_counts().to_frame().rename({col:"Misclassified"}, axis=1)/df_misclass[col].value_counts().sum()
            values = corr_value_counts.join(wrong_value_counts, how="outer").fillna(0)
            # display(values)
            values.plot.barh(ax=ax)
        
        elif df_correct[col].dtype == "float":
            corr_value_counts = df_correct[col].rename("Correct")#/len(df_correct[col])
            wrong_value_counts = df_misclass[col].rename("Misclassified")#/len(df_misclass[col])
#             display(corr_value_counts)
#             display(wrong_value_counts)
            
            wrong_value_counts.plot(
                kind="hist", 
                logy=logy,
                histtype='step', 
                ax=ax,
                density=True,
            legend=True)
            corr_value_counts.plot(
                kind="hist", 
                logy=logy,
                histtype='step', 
                ax=ax,
                density=True,
            legend=True)
            
        else:
            raise ValueError(f"unsupported dtype: {df_correct[col].dtype}")
        
        ax.set_title(f'{col}', fontsize=16, weight="bold")
        ax.set_ylabel('Frequency', fontsize=14)
        ax.tick_params(axis='both',labelsize=13)
        
    fig.subplots_adjust(top=title_adjustment)    
            
def plot_missclassified_hists_by_strata(df_correct, df_misclass, cols,count_threshold=20, logy=True, figsize=(8,15)):
    
    ncols = 2
    fig, axs = plt.subplots(nrows=int(np.round(len(cols)/2)), ncols=ncols, figsize=figsize)
    
    for col, ax in zip(cols, axs.ravel()):
        is_counts = True
        try:
            df_correct[col].astype(pd.Int64Dtype()) 
            df_misclass[col].astype(pd.Int64Dtype()) 
            # if there are more than a certain number of values, it's probably not a count
            if len(df_correct[col].value_counts()) > count_threshold:
                is_counts = False
#             ignore age, since they're all integers but it's not a count variable                
            if col == "age":
                is_counts = False
        except:
            is_counts = False
            print(col)
        
        if df_correct[col].dtype == "object" or is_counts:
            corr_value_counts = df_correct[col].value_counts().to_frame().rename({col:"Correct"}, axis=1)/df_correct[col].value_counts().sum()
            wrong_value_counts = df_misclass[col].value_counts().to_frame().rename({col:"Misclassified"}, axis=1)/df_misclass[col].value_counts().sum()
            values = corr_value_counts.join(wrong_value_counts, how="outer").fillna(0)
            # display(values)
            values.plot.barh(ax=ax)
        
        elif df_correct[col].dtype == "float":
            corr_value_counts = df_correct[col].rename("Correct")#/len(df_correct[col])
            wrong_value_counts = df_misclass[col].rename("Misclassified")#/len(df_misclass[col])
#             display(corr_value_counts)
#             display(wrong_value_counts)
            
            wrong_value_counts.plot(
                kind="hist", 
                logy=logy,
                histtype='step', 
                ax=ax,
                density=True,
            legend=True)
            corr_value_counts.plot(
                kind="hist", 
                logy=logy,
                histtype='step', 
                ax=ax,
                density=True,
            legend=True)
            
        else:
            raise ValueError(f"unsupported dtype: {df_correct[col].dtype}")
        
        ax.title.set_text(f'{col}')        