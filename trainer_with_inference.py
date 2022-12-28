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
spacy.require_gpu()

@hydra.main(config_path="conf/", config_name="config")
def main(cfg: DictConfig) -> None:
    
    print(OmegaConf.to_yaml(cfg))
    
    # some checks on config:
    if cfg.train.add_ner_so and not cfg.train.so_sections:
        raise Exception("Tried to add NER into S&O without including those sections")

    
    if cfg.hardware.gpu:
        print(f"Has cuda: {torch.cuda.is_available()}")
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(device)
    
    print(f"Using huggingface transformer model: {cfg.model.model_name}")
    # Define file paths
    
    if cfg.model.model_type == "bert" or cfg.model.model_type == "roberta":
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name)
        if cfg.model.model_type == "roberta":
            model = RobertaForSequenceClassification.from_pretrained(cfg.model.model_name, num_labels=4)
        elif cfg.model.model_type == "bert":
            model = BertForSequenceClassification.from_pretrained(cfg.model.model_name, num_labels=4)
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
        "Relation":Value("string"),
        "S":Value("string"),
        "O":Value("string")        
        
    }) 

    dataset = load_dataset("csv", data_files={
                                "train":cfg.data.n2c2_data_dir + "train_so.csv",
                                "valid":cfg.data.n2c2_data_dir + "dev_so.csv",
                                # "train":cfg.data.n2c2_data_dir + "train.csv",
                                # "valid":cfg.data.n2c2_data_dir + "dev.csv",
        
                            },
                           features=features)
    test_features = Features({
        'ROW ID':Value("int64"),
        'HADM ID':Value("int64"),
        'Assessment':Value("string"),
        'Plan Subsection':Value("string"),
        "S":Value("string"),
        "O":Value("string")        
        
    }) 

    test_dataset = load_dataset("csv", data_files={
                                # "test":cfg.data.n2c2_data_dir + "n2c2_test_noLabel.csv",
                                "test":cfg.data.n2c2_data_dir + "n2c2_track3_test_so.csv",
                            },
                           features=test_features)
    

    if cfg.train.fast_dev_run:
        dataset['train'] = dataset['train'].shard(num_shards=1000, index=0)
        dataset['valid'] = dataset['valid'].shard(num_shards=50, index=0)

    # create encoded class labels and rename
    dataset = dataset.class_encode_column("Relation")
    label2id = {'Not Relevant':3, 'Neither':2, 'Indirect':1, 'Direct':0}
    id2label = {v:k for k,v in label2id.items()}
    dataset = dataset.align_labels_with_mapping(label2id, "Relation")
    dataset = dataset.rename_column("Relation", "label")
    
    # drop symptom list at beginning of some assessments
    # dataset['train'] = dataset['train'].map(split_leading_symptom_list)
    # dataset['valid'] = dataset['valid'].map(split_leading_symptom_list)
    if cfg.train.so_sections:
        dataset = dataset.map(partial(add_SO_sections))
        test_dataset = test_dataset.map(partial(add_SO_sections))
    
    print("AFTER TRAIN _SO sections")
    print(dataset['valid'][0])
    print(test_dataset['test'][0])
    
    if cfg.train.add_ner:
        nlp_assessment = spacy.load(cfg.pretrained.spacy_assessment, exclude="parser")
        nlp_plan = spacy.load(cfg.pretrained.spacy_plan, exclude="parser")
        
        # add the named entities
        dataset['train'] = dataset['train'].map(partial(add_ner_assessment, nlp=nlp_assessment))
        dataset['train'] = dataset['train'].map(partial(add_ner_plan, nlp=nlp_plan))        

        dataset['valid'] = dataset['valid'].map(partial(add_ner_assessment, nlp=nlp_assessment))
        dataset['valid'] = dataset['valid'].map(partial(add_ner_plan, nlp=nlp_plan))        

        test_dataset['test'] = test_dataset['test'].map(partial(add_ner_assessment, nlp=nlp_assessment))
        test_dataset['test'] = test_dataset['test'].map(partial(add_ner_plan, nlp=nlp_plan))        
        
        # we ASSUME that the ner labels we want are lowercase, UNLIKE the standard ones in the model
        spans = [x for x in nlp_plan.get_pipe("ner").labels if x.islower()] + [x for x in nlp_assessment.get_pipe("ner").labels if x.islower()]

        tokens = []
        for span in spans:
            tokens.append("<" + span + ">")
            tokens.append("</" + span + ">")            
        
        # add the span tags to the vocab
        _ = tokenizer.add_tokens(tokens)
        model.resize_token_embeddings(len(tokenizer))
    
    elif cfg.train.add_ner_end:
        nlp_assessment = spacy.load(cfg.pretrained.spacy_assessment, exclude="parser")
        nlp_plan = spacy.load(cfg.pretrained.spacy_plan, exclude="parser")
        
        # add the named entities
        dataset['train'] = dataset['train'].map(partial(add_ner_assessment_end, nlp=nlp_assessment))
        dataset['train'] = dataset['train'].map(partial(add_ner_plan_end, nlp=nlp_plan))        

        dataset['valid'] = dataset['valid'].map(partial(add_ner_assessment_end, nlp=nlp_assessment))
        dataset['valid'] = dataset['valid'].map(partial(add_ner_plan_end, nlp=nlp_plan))        

        test_dataset['test'] = test_dataset['test'].map(partial(add_ner_assessment_end, nlp=nlp_assessment))
        test_dataset['test'] = test_dataset['test'].map(partial(add_ner_plan_end, nlp=nlp_plan))        
        
        # we ASSUME that the ner labels we want are lowercase, UNLIKE the standard ones in the model
        spans = [x for x in nlp_plan.get_pipe("ner").labels if x.islower()] + [x for x in nlp_assessment.get_pipe("ner").labels if x.islower()]

        tokens = []
        for span in spans:
            tokens.append("</" + span + ">")            
        
        # add the span tags to the vocab
        _ = tokenizer.add_tokens(tokens)
        model.resize_token_embeddings(len(tokenizer))
        
    
    if cfg.train.add_ner_so:
        nlp_so = spacy.load(cfg.pretrained.spacy_so, exclude="parser")
        
        # add the named entities
        dataset['train'] = dataset['train'].map(partial(add_ner_so, nlp=nlp_so))
        dataset['valid'] = dataset['valid'].map(partial(add_ner_so, nlp=nlp_so))
        test_dataset['test'] = test_dataset['test'].map(partial(add_ner_so, nlp=nlp_so))
        
        # we ASSUME that the ner labels we want are lowercase, UNLIKE the standard ones in the model
        # we're assuming that they don't get added twice, if they've been added above with the assessment/plan models
        spans = [x for x in add_ner_so.get_pipe("ner").labels if x.islower()]

        tokens = []
        for span in spans:
            tokens.append("<" + span + ">")
            tokens.append("</" + span + ">")            
        
        # add the span tags to the vocab
        _ = tokenizer.add_tokens(tokens)
        model.resize_token_embeddings(len(tokenizer))
    
    if cfg.train.drop_mimic_deid:
        dataset = dataset.map(remove_mimic_deid)
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
        
        dataset = dataset.map(partial(expand_abbreviations, spacy_pip=abbv_nlp, abbv_map=med_abbvs, unq_sfs=unq_sfs))
        test_dataset = test_dataset.map(partial(expand_abbreviations, spacy_pip=abbv_nlp, abbv_map=med_abbvs, unq_sfs=unq_sfs))
                                  
    
    # create training args and Trainer
    if cfg.save.save_model:
        save_strategy = "epoch"
        save_total_limit = 1
        load_best_model_at_end = True
    else:
        save_strategy = "no"
        save_total_limit = 0    
        load_best_model_at_end = False

    training_args = TrainingArguments(output_dir=f"./outputs/{cfg.model.model_id}", 
                                        overwrite_output_dir=False,
                                        evaluation_strategy="epoch",
                                        learning_rate=1e-5,
                                        load_best_model_at_end=load_best_model_at_end,
                                        warmup_ratio = 0.06,
                                        gradient_accumulation_steps = 8,
                                        num_train_epochs=cfg.train.epochs,
                                        per_device_train_batch_size=cfg.train.batch_size,
                                        fp16=True,
                                        gradient_checkpointing=True,
                                        save_total_limit=save_total_limit,
                                        save_strategy=save_strategy,
                                        log_level="debug",
                                        logging_dir=f"./outputs/test",
                                        logging_strategy="steps",
                                        logging_first_step=True,
                                        logging_steps=500,                                      
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
    dataset = dataset.map(partial(tokenize_function, tokenizer=tokenizer), batched=True)
    test_dataset = test_dataset.map(partial(tokenize_function, tokenizer=tokenizer), batched=True)    
    
    print(tokenizer.decode(dataset['valid'][0]['input_ids']))
    print(tokenizer.decode(test_dataset['test'][0]['input_ids']))
        
    # cast as pytorch tensors and select a subset of columns we want
    if cfg.model.model_type == "gptneo":
        dataset['train'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        dataset['valid'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        test_dataset['test'].set_format(type='torch', columns=['input_ids', 'attention_mask'])
    elif cfg.model.model_type == "roberta":
        dataset['train'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        dataset['valid'].set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])    
        test_dataset['test'].set_format(type='torch', columns=['input_ids', 'attention_mask'])    
    else:
        dataset['train'].set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
        dataset['valid'].set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])    
        test_dataset['test'].set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask'])
    
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
            train_dataset=dataset['train'],
            eval_dataset=dataset['valid'],
            compute_metrics=partial(compute_metrics, metric_dict=metric_dict),
            data_collator=data_collator,        
        )
    else:
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
    
    # predict for metrics
    metrics = trainer.evaluate(dataset['valid'])
    
    fpr, tpr, roc_auc = metrics['eval_roc']
    precision, recall, ap = metrics['eval_pr']
    
    print("id2label", id2label)
    plot_multiclass_roc(fpr, tpr, roc_auc, figsize=(8, 6), labels=id2label, fname="Eval_AUROC.png")
    plot_multiclass_pr(precision, recall, ap, figsize=(8, 6), labels=id2label, fname="Eval_AUPRC.png")
    
    predict_output = trainer.predict(test_dataset['test'])
    preds = np.argmax(predict_output.predictions, axis=-1)
    
    test_dataset['test'].to_csv("test_dataset_output.csv")
    np.savetxt("test_predictions.csv", preds, delimiter=",")
    # np.savetxt("test_label_ids.csv", predict_output.label_ids, delimiter=",")
    
    
if __name__ == "__main__":
    main()
    
