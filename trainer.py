import re 
from functools import partial

import pandas as pd 
import numpy as np

import torch 
import spacy
import sklearn

from datasets import load_dataset
from datasets import Value, ClassLabel, Features, DatasetDict
from datasets import load_metric

import transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification
from transformers import GPT2Tokenizer, GPTNeoForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import logging
from transformers import TrainingArguments, Trainer

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
                                                                pad_token_id=tokenizer.convert_tokens_to_ids("[PAD]"))
        model.resize_token_embeddings(len(tokenizer))        
    else:
        raise ValueError(f"Model type isn't bert or gpt-neo, it's {cfg.model.model_type}")

    # Read MIMIC notes
    # notes = pd.read_csv(cfg.data.mimic_data_dir + "NOTEEVENTS.csv")

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
    # dataset['train'] = dataset['train'].map(split_leading_symptom_list)
    # dataset['valid'] = dataset['valid'].map(split_leading_symptom_list)
    
    if cfg.train.add_ner:
        nlp_assessment = spacy.load(cfg.pretrained.spacy_assessment, exclude="parser")
        nlp_plan = spacy.load(cfg.pretrained.spacy_plan, exclude="parser")
        
        # add the named entities
        dataset['train'] = dataset['train'].map(partial(add_ner_assessment, nlp=nlp_assessment))
        dataset['train'] = dataset['train'].map(partial(add_ner_plan, nlp=nlp_plan))        

        dataset['valid'] = dataset['valid'].map(partial(add_ner_assessment, nlp=nlp_assessment))
        dataset['valid'] = dataset['valid'].map(partial(add_ner_plan, nlp=nlp_plan))        
        
        # we ASSUME that the ner labels we want are lowercase, UNLIKE the standard ones in the model
        spans = [x for x in nlp_plan.get_pipe("ner").labels if x.islower()] + [x for x in nlp_assessment.get_pipe("ner").labels if x.islower()]

        tokens = []
        for span in spans:
            tokens.append("<" + span + ">")
            tokens.append("</" + span + ">")            
        
        # add the span tags to the vocab
        _ = tokenizer.add_tokens(tokens)
        model.resize_token_embeddings(len(tokenizer))
        
    if cfg.train.drop_mimic_deid:
        dataset = dataset.map(remove_mimic_deid)
        
        
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
    
    # tokenize
    dataset = dataset.map(partial(tokenize_function, tokenizer=tokenizer), batched=True)
    
    print(tokenizer.decode(dataset['valid'][0]['input_ids']))
        
    # cast as pytorch tensors and select a subset of columns we want
    if cfg.model.model_type == "gptneo":
        dataset['train'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        dataset['valid'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    else:
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
        compute_metrics=partial(compute_metrics, metric_dict=metric_dict),
        data_collator=data_collator,
    )    
    
    # train!!
    trainer.train()    

if __name__ == "__main__":
    main()
    
