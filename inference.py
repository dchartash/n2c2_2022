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
import evaluate


import transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification
from transformers import GPT2Tokenizer, GPTNeoForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import logging
from transformers import TrainingArguments, Trainer

from transformers import RobertaForSequenceClassification, BertForSequenceClassification

from omegaconf import DictConfig, OmegaConf
import hydra

from preprocessing.cleaning_utils import *
from train_utils.metrics import *
from train_utils.plot_utils import *
from train_utils.custom_trainer import *

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

logging.set_verbosity_warning()


@hydra.main(config_path="conf/", config_name="config")
def main(cfg: DictConfig) -> None:
    
    print(OmegaConf.to_yaml(cfg))
    
    if cfg.hardware.gpu:
        print(f"Has cuda: {torch.cuda.is_available()}")
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(device)
    
    # Define file paths
    
    if cfg.model.model_type == "bert" or cfg.model.model_type == "roberta":
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
        if cfg.model.model_type == "roberta":
            model = RobertaForSequenceClassification.from_pretrained(cfg.model.trained_model_path, num_labels=4, 
                                                                     local_files_only=True)
            print(f"Using huggingface transformer model: RoBERTa")
            
        elif cfg.model.model_type == "bert":
            model = BertForSequenceClassification.from_pretrained(cfg.model.trained_model_path, num_labels=4, 
                                                                     local_files_only=True)
            print(f"Using huggingface transformer model: BERT")            
    elif cfg.model.model_type == "gptneo":    
        tokenizer = GPT2Tokenizer.from_pretrained(cfg.model.trained_model_path, local_files_only=True)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model = GPTNeoForSequenceClassification.from_pretrained(cfg.model.trained_model_path, num_labels=4,
                                                                local_files_only=True,
                                                                problem_type="single_label_classification",
                                                                pad_token_id=tokenizer.convert_tokens_to_ids("[PAD]"))
        model.resize_token_embeddings(len(tokenizer))        
        print(f"Using huggingface transformer model: GPT-2")
    else:
        raise ValueError(f"Model type isn't bert or gpt-neo, it's {cfg.model.model_type}")

    # Read MIMIC notes
    # notes = pd.read_csv(cfg.data.mimic_data_dir + "NOTEEVENTS.csv")

    # create hf Dataset
    classes = ['Not Relevant', 'Neither', 'Indirect', 'Direct']
    
    # instead we will use the raw text for now
    test_features = Features({
        'ROW ID':Value("int64"),
        'HADM ID':Value("int64"),
        'Assessment':Value("string"),
        'Plan Subsection':Value("string"),
        "Relation":Value("string")
    }) 

    test_dataset = load_dataset("csv", data_files={
                                "test":cfg.data.n2c2_data_dir + "n2c2_track3_test.csv",
                            },
                           features=test_features)
    

    if cfg.train.fast_dev_run:
        # dataset['train'] = dataset['train'].shard(num_shards=2000, index=0)
        test_dataset['test'] = test_dataset['test'].shard(num_shards=20, index=0)        
    #     dataset['valid'] = dataset['valid'].shard(num_shards=50, index=0)

    # create encoded class labels and rename
    test_dataset = test_dataset.class_encode_column("Relation")
    label2id = {'Not Relevant':3, 'Neither':2, 'Indirect':1, 'Direct':0}
    id2label = {v:k for k,v in label2id.items()}
    test_dataset = test_dataset.align_labels_with_mapping(label2id, "Relation")
    test_dataset = test_dataset.rename_column("Relation", "label")
    
    
    if cfg.train.add_ner:
        nlp_assessment = spacy.load(cfg.pretrained.spacy_assessment, exclude="parser")
        nlp_plan = spacy.load(cfg.pretrained.spacy_plan, exclude="parser")
        
        # add the named entities
        test_dataset['test'] = test_dataset['test'].map(partial(add_ner_assessment, nlp=nlp_assessment))
        test_dataset['test'] = test_dataset['test'].map(partial(add_ner_plan, nlp=nlp_plan))        
        
        # we ASSUME that the ner labels we want are lowercase, UNLIKE the standard ones in the model
        spans = [x for x in nlp_plan.get_pipe("ner").labels if x.islower()] + [x for x in
                                                                               nlp_assessment.get_pipe("ner").labels 
                                                                               if x.islower()]

        tokens = []
        for span in spans:
            tokens.append("<" + span + ">")
            tokens.append("</" + span + ">")            
        
        # add the span tags to the vocab
        _ = tokenizer.add_tokens(tokens)
        model.resize_token_embeddings(len(tokenizer))
        
    if cfg.train.drop_mimic_deid:
        test_dataset = test_dataset.map(remove_mimic_deid)
       
    if cfg.train.expand_abbvs:
        abbv_nlp = spacy.load("en_core_sci_lg")
        abbreviations = pd.read_csv(cfg.data.abbreviation_inventory, sep="|", na_filter=False)
        med_abbvs = abbreviations[abbreviations['Source'].isin(["Vanderbilt Clinic Notes", "Vanderbilt Discharge Sums", "Berman", "Stetson",   
                                                                "Columbia"])]
        med_abbvs = med_abbvs[~med_abbvs['SF'].isin(abbv_nlp.Defaults.stop_words)]
        med_abbvs = med_abbvs[~med_abbvs['SF'].isin(["man", "woman", "old", "Mr.", "Ms.", "Mrs", "M", "F"])]
        med_abbvs = med_abbvs.astype({"Source":"category"})
        sorter = ["Vanderbilt Discharge Sums", "Vanderbilt Clinic Notes",  "Stetson", "Columbia", "Berman"]
        med_abbvs.Source.cat.set_categories(sorter, inplace=True)
        med_abbvs = med_abbvs.sort_values(['Source'])
        unq_sfs = med_abbvs['SF'].unique()
        
        test_dataset = test_dataset.map(partial(expand_abbreviations, spacy_pip=abbv_nlp, abbv_map=med_abbvs, 
                                                unq_sfs=unq_sfs))
    
    # create training args and Trainer
    training_args = TrainingArguments(output_dir=f"./outputs/{cfg.model.model_id}", 
                                        overwrite_output_dir=False,
                                        evaluation_strategy="epoch",
                                        learning_rate=1e-5,
                                        load_best_model_at_end=True,
                                        warmup_ratio = 0.06,
                                        gradient_accumulation_steps = 8,
                                        num_train_epochs=cfg.train.epochs,
                                        per_device_train_batch_size=cfg.train.batch_size,
                                        fp16=True,
                                        gradient_checkpointing=True,
                                        save_total_limit = 1,
                                        save_strategy="epoch",
                                        # save_steps=1000,
    )
    
    # metrics to track
    acc = load_metric("accuracy")
    macrof1 = load_metric("f1")    
    roc_auc_score = evaluate.load("roc_auc", "multiclass")

    # create metric_dict for compute_metrics
    metric_dict = {}
    metric_dict['accuracy'] = {"metric":acc}
    metric_dict['f1-macro'] = {"metric":macrof1, "average":"macro"}
    metric_dict['auroc'] = {'metric':roc_auc_score, "multi_class":'ovr'}
    metric_dict['roc'] = {}
    metric_dict["pr"] = {}
    # tokenize
    test_dataset = test_dataset.map(partial(tokenize_function, tokenizer=tokenizer), batched=True)    
    
    print("DECODED: ", tokenizer.decode(test_dataset['test'][0]['input_ids']))
        
    # cast as pytorch tensors and select a subset of columns we want
    if cfg.model.model_type == "gptneo":
        test_dataset['test'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    elif cfg.model.model_type == "roberta":
        test_dataset['test'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    else:
        test_dataset['test'].set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
    
    # create collator
    data_collator = DataCollatorWithPadding(tokenizer,
                                            max_length=512, 
                                            padding="longest",
                                            return_tensors="pt")    
    # create Trainer
    if cfg.train.weighted_loss:
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            eval_dataset=test_dataset['test'],            
            compute_metrics=partial(compute_metrics, metric_dict=metric_dict),
            data_collator=data_collator,        
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=test_dataset['test'],            
            compute_metrics=partial(compute_metrics, metric_dict=metric_dict),
            data_collator=data_collator,
        )    
                
    # predict for metrics
    metrics = trainer.evaluate(test_dataset['test'], metric_key_prefix="test")
    
    print(metrics)
    fpr, tpr, roc_auc = metrics['test_roc']
    precision, recall, ap = metrics['test_pr']
    
    print("id2label", id2label)
    plot_multiclass_roc(fpr, tpr, roc_auc, figsize=(8, 6), labels=id2label, fname="Test_AUROC.png")
    plot_multiclass_pr(precision, recall, ap, figsize=(8, 6), labels=id2label, fname="Test_AUPRC.png")
    
#     predict_output = trainer.predict(test_dataset['test'])
#     preds = np.argmax(predict_output.predictions, axis=-1)
    
#     test_dataset['test'].to_csv("test_dataset_output.csv")
#     np.savetxt("test_predictions.csv", preds, delimiter=",")
    # np.savetxt("test_label_ids.csv", predict_output.label_ids, delimiter=",")
    
    
if __name__ == "__main__":
    main()
    
