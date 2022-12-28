import datasets
import numpy as np
import torch 
from sklearn.preprocessing import label_binarize

from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    roc_curve,
    precision_recall_curve,
    auc,
    average_precision_score,
)

def compute_metrics(eval_pred, metric_dict):
    '''
    The metric_dict will look like: {"metric name": {"metric":metric,
                                                     **kwargs}
    '''
    logits, labels = eval_pred
    
    predictions = np.argmax(logits, axis=-1)
    probs = torch.nn.functional.softmax(torch.tensor(logits).float(), dim=-1)
    metric_results = {}
    
    for metric_name, metric_args in metric_dict.items():
        
        if metric_name == "auroc":
            metric = metric_args["metric"]
            metric_results[metric_name] = metric.compute(prediction_scores=probs, references=labels, 
                                                         **{key:val for key,val in metric_args.items() if key != "metric"})
        elif metric_name == "roc":
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            y = label_binarize(labels, classes=[0, 1, 2, 3])
            for i in range(4):
                fpr_v, tpr_v, _ = roc_curve(y[:, i], probs[:, i])
                fpr[i] = fpr_v.tolist()
                tpr[i] = tpr_v.tolist()                
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            metric_results[metric_name] = (fpr, tpr, roc_auc)
        elif metric_name == "pr":
            precision = dict()
            recall = dict()
            ap = dict()

            y = label_binarize(labels, classes=[0, 1, 2, 3])
            ap_v = average_precision_score(y, probs, average=None)
            
            for i in range(4):
                p_v, r_v, _ = precision_recall_curve(y[:, i], probs[:, i])

                precision[i] = p_v.tolist()
                recall[i] = r_v.tolist()   
                ap[i] = ap_v[i]
                # roc_auc[i] = auc(fpr[i], tpr[i])
            
            metric_results[metric_name] = (precision, recall, ap)
            
        else:
            metric = metric_args["metric"]
            metric_results[metric_name] = metric.compute(predictions=predictions, references=labels, 
                                                         **{key:val for key,val in metric_args.items() if key != "metric"})    
    
    return metric_results
