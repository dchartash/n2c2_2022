import pandas as pd 
import numpy as np

from preprocessing.cleaning_utils import *
from functools import partial
import torch 

from datasets import load_dataset
from datasets import Value, ClassLabel, Features, DatasetDict
from datasets import load_metric

import transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import logging
from transformers import TrainingArguments, Trainer

logging.set_verbosity_warning()


def main():

    # Read MIMIC notes
    notes = pd.read_csv(mimic_dir + "NOTEEVENTS.csv")

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
                                "train":n2c2_data_dir + "train.csv",
                                "valid":n2c2_data_dir + "dev.csv",
                            },

                           features=features)

    # create encoded class labels and rename
    dataset = dataset.class_encode_column("Relation")
    dataset = dataset.rename_column("Relation", "label")
    
    # create training args and Trainer
    training_args = TrainingArguments(output_dir="test_trainer", 
                                      evaluation_strategy="epoch",

    )
    
    # metrics to track
    acc = load_metric("accuracy")
    macrof1 = load_metric("f1")    
    
    # tokenize
    dataset = dataset.map(tokenize_function, batched=True)
    
    print(bio_clinicalbert_tokenizer.decode(dataset['valid'][52]['input_ids']))
        
    # cast as pytorch tensors and select a subset of columns we want
    dataset['train'].set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
    dataset['valid'].set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])    
    
    # create collator
    data_collator = DataCollatorWithPadding(tokenizer,
                                            max_length=512, 
                                            padding="max_length",
                                            return_tensors="pt")    
    # create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['valid'],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )    
    
    # train!!
    trainer.train()    
    
    
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy":acc.compute(predictions=predictions, references=labels),
            "f1-macro":macrof1.compute(predictions=predictions, references=labels, 
                                       average="macro")}
if __name__ == "__main__":
    print(torch.cuda_is_available())
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    
    # Define file paths
    mimic_data_dir = "/home/vs428/project/MIMIC/files/mimiciii/1.4/"
    n2c2_data_dir = "/home/vs428/project/n2c2/2022/Data/"
    hf_modelhub_path = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(hf_modelhub_path)
    model = AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=4)
    main()
    