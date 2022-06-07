import datasets
import numpy as np

def compute_metrics(eval_pred, metric_dict):
    '''
    The metric_dict will look like: {"metric name": {"metric":metric,
                                                     **kwargs}
    '''
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric_results = {}
    for metric_name, metric_args in metric_dict.items():
        metric = metric_args["metric"]
        
        metric_results[metric_name] = metric.compute(predictions=predictions, references=labels, **{key:val for key,val in metric_args.items() if key != "metric"})    
    
    return metric_results
