import pandas as pd 
import numpy as np
import tqdm

from functools import partial
import torch 
from torch.utils.data import Dataset

from datasets import load_dataset
from datasets import Value, ClassLabel, Features, DatasetDict
from datasets import load_metric


import transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification
from transformers import GPT2Tokenizer, GPTNeoForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import logging
from transformers import TrainingArguments, Trainer
from transformers import pipeline
from transformers import EvalPrediction


from omegaconf import DictConfig, OmegaConf
import hydra

from preprocessing.cleaning_utils import *
from train_utils.metrics import *

logging.set_verbosity_warning()


@hydra.main(config_path="conf/", config_name="config")
def main(cfg: DictConfig) -> None:
    
    print(OmegaConf.to_yaml(cfg))
    
    if cfg.hardware.gpu:
        print(f"Has cuda: {torch.cuda.is_available()}")
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(device)
    
    print(f"Using huggingface transformer model: {cfg.model.model_name}")
    # Define file paths
    
    if cfg.model.model_type == "bert":
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(cfg.model.model_name, num_labels=4)
    elif cfg.model.model_type == "gptneo":    
        tokenizer = GPT2Tokenizer.from_pretrained(cfg.model.model_name)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        model = GPTNeoForSequenceClassification.from_pretrained(cfg.model.model_name, num_labels=4,
                                                                problem_type="single_label_classification",
                                                                pad_token_id=tokenizer.convert_tokens_to_ids("[PAD]"),
)
        model.resize_token_embeddings(len(tokenizer))        
    else:
        raise ValueError(f"Model type isn't bert or gpt-neo, it's {cfg.model.model_type}")

    # create hf Dataset
    classes = ['Not Relevant', 'Neither', 'Indirect', 'Direct']
    
    # instead we will use the raw text for now
    features = Features({
        'ROW ID':Value("int64"),
        'HADM ID':Value("int64"),
        'Assessment':Value("string"),
        'Plan Subsection':Value("string"),
        "Relation":Value("string")
    }) 

    dataset = load_dataset("csv", data_files={
                                "train":cfg.data.n2c2_data_dir + "train.csv",
                                "valid":cfg.data.n2c2_data_dir + "dev.csv",
                            },
                           features=features)

    if cfg.train.fast_dev_run:
        dataset['train'] = dataset['train'].shard(num_shards=1000, index=0)
        dataset['valid'] = dataset['train'].shard(num_shards=100, index=0)

    # create encoded class labels and rename
    dataset = dataset.class_encode_column("Relation")
    dataset = dataset.rename_column("Relation", "label")
    
    # drop symptom list at beginning of some assessments
    dataset['train'] = dataset['train'].map(split_leading_symptom_list)
    dataset['valid'] = dataset['valid'].map(split_leading_symptom_list)
    
    # create training args and Trainer
    training_args = TrainingArguments(output_dir="test_trainer", 
                                      evaluation_strategy="epoch",
                                      num_train_epochs=cfg.train.epochs,
                                      per_device_train_batch_size=4,
                                      
    )
    
    # metrics to track
    acc = load_metric("accuracy")
    macrof1 = load_metric("f1")    

    # create metric_dict for compute_metrics
    metric_dict = {}
    metric_dict['accuracy'] = {"metric":acc}
    metric_dict['f1-macro'] = {"metric":macrof1, "average":"macro"}
    
    print("Creating GPT prompts...")
    # create GPT prompts
    prompts = []
    labels = []
    for data in dataset['valid']:
        question = 'To which category does the text belong?: "Direct", "Indirect", "Neither", "Not Relevant"'
        prompt_str = question + '\nAssessment: ' + data['Assessment'] + '\nPlan: ' + data['Plan Subsection'] + '\nLabel: ' #\
            # + str(data['label'])
        labels.append(data['label'])
        prompts.append(prompt_str)
    
    class ListDataset(Dataset):
        def __init__(self, original_list):
            self.original_list = original_list

        def __len__(self):
            return len(self.original_list)

        def __getitem__(self, i):
            return self.original_list[i]    
        
    print("Creating Dataset obj...")        
    mydataset = ListDataset(prompts)

    print("Defining pipeline...")
    rel_classifier = pipeline(task="text-classification", model=model, tokenizer=tokenizer, 
                              return_all_scores=True, device=0, padding="max_length", max_length=512,
                             truncation=True)
    
    logits = np.zeros(shape=(len(prompts),4))
    labels = np.array(labels)

    print("Inference...")    
    for idx, pred in enumerate(tqdm.tqdm(rel_classifier(mydataset))):
        pred_logits = [x['score'] for x in pred]
        logits[idx, :] = pred_logits
                
        
    compute_metrics_func=partial(compute_metrics, metric_dict=metric_dict)
    metrics = compute_metrics_func(EvalPrediction(logits, labels))
    print("Metrics:", metrics)
    
if __name__ == "__main__":
    main()
    