#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

from preprocessing.cleaning_utils import *
from train_utils.metrics import *
from train_utils.plot_utils import *
from train_utils.custom_trainer import *

import matplotlib.pyplot as plt
import seaborn as sns

from train_utils.metrics import *
from transformers import RobertaForSequenceClassification
from transformers import TrainingArguments, Trainer


# In[2]:


import pandas as pd
from datasets import load_dataset
from datasets import Value, ClassLabel, Features, DatasetDict


# In[3]:


# pd.read_csv("/home/vs428/project/n2c2/2022/Data/n2c2_track3_test.csv")
# data = pd.read_csv()
DATA_DIR = "/home/vs428/project/n2c2/2022/Data/dev.csv"
OUTPUT_DIR = "/home/vs428/Documents/n2c2_2022/outputs/2022-10-26/03-12-17/outputs/roberta_large/checkpoint-432"
BATCH_SIZE = 16
SPACY_ASSESSMENT =  "/home/vs428/project/n2c2_spacy_models/assessment_model_v2/model-best"
SPACY_PLAN =  "/home/vs428/project/n2c2_spacy_models/plan_subsection_model_v3/model-best"


# In[64]:


model = RobertaForSequenceClassification.from_pretrained(OUTPUT_DIR,
                                                        num_labels=4)
tokenizer = AutoTokenizer.from_pretrained("/home/vs428/project/models/RoBERTa-large-PM-M3-Voc/RoBERTa-large-PM-M3-Voc-hf")


# In[65]:


nlp_assessment = spacy.load(SPACY_ASSESSMENT, exclude="parser")
nlp_plan = spacy.load(SPACY_PLAN, exclude="parser")


# In[109]:


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
                            "train":DATA_DIR,
                        },
                       features=features)

# create encoded class labels and rename
dataset = dataset.class_encode_column("Relation")
label2id = {'Not Relevant':3, 'Neither':2, 'Indirect':1, 'Direct':0}
id2label = {v:k for k,v in label2id.items()}
dataset = dataset.align_labels_with_mapping(label2id, "Relation")
dataset = dataset.rename_column("Relation", "label")


# In[110]:


dataset = dataset.shuffle(seed=42)


# In[111]:


dataset = dataset['train']#.select(range(100))


# In[112]:


# dataset[32]


# In[113]:


# add the named entities
dataset = dataset.map(partial(add_ner_assessment, nlp=nlp_assessment))
dataset = dataset.map(partial(add_ner_plan, nlp=nlp_plan))        


# In[114]:


# we ASSUME that the ner labels we want are lowercase, UNLIKE the standard ones in the model
spans = [x for x in nlp_plan.get_pipe("ner").labels if x.islower()] + [x for x in nlp_assessment.get_pipe("ner").labels if x.islower()]

tokens = []
for span in spans:
    tokens.append("<" + span + ">")
    tokens.append("</" + span + ">")            

# add the span tags to the vocab
_ = tokenizer.add_tokens(tokens)
model.resize_token_embeddings(len(tokenizer))


# In[115]:


# create training args and Trainer
test_args = TrainingArguments(output_dir=OUTPUT_DIR, 
                                    do_train = False,
                                    do_predict = True,
                                    evaluation_strategy="steps",
                                    learning_rate=1e-5,
                                    load_best_model_at_end=True,
                                    warmup_ratio = 0.06,
                                    gradient_accumulation_steps = 8,
                                    per_device_eval_batch_size = BATCH_SIZE,   
                                    dataloader_drop_last = False,  
                                    fp16=True,
                                    log_level="debug",
                                    logging_dir=f"./outputs/test",
                                    logging_strategy="steps",
                                    logging_first_step=True,
                                    logging_steps=500,                                      
                                    # save_steps=1000,
)


# In[116]:


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


# In[117]:


# tokenize
dataset = dataset.map(partial(tokenize_function, tokenizer=tokenizer), batched=True)
print(tokenizer.decode(dataset[0]['input_ids']))



# In[118]:


# cast as pytorch tensors and select a subset of columns we want

dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# create collator
data_collator = DataCollatorWithPadding(tokenizer,
                                        max_length=512, 
                                        padding="longest",
                                        return_tensors="pt")    
# create Trainer
trainer = Trainer(
    model=model,
    args=test_args,
    compute_metrics=partial(compute_metrics, metric_dict=metric_dict),
    data_collator=data_collator,
)    

# predict for metrics
metrics = trainer.predict(dataset)


# In[119]:


fpr, tpr, roc_auc = metrics[2]['test_roc']
precision, recall, ap = metrics[2]['test_pr']


# In[120]:


print(roc_auc), print(ap)


# In[121]:


print("id2label", id2label)
# plot_multiclass_roc(fpr, tpr, roc_auc, figsize=(8, 6), labels=id2label, fname="Eval_AUROC.png")
# plot_multiclass_pr(precision, recall, ap, figsize=(8, 6), labels=id2label, fname="Eval_AUPRC.png")

preds = np.argmax(metrics.predictions, axis=-1)

# test_dataset['test'].to_csv("test_dataset_output.csv")
# np.savetxt("test_predictions.csv", preds, delimiter=",")
# np.savetxt("test_label_ids.csv", predict_output.label_ids, delimiter=",")


# In[122]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[123]:


y_true = dataset['label'].numpy()

# In[124]:


indirect_wrongs = np.where((y_true == 1) & (y_true != preds))


TSET_IDX = 35
y_true[TSET_IDX], preds[TSET_IDX], dataset['Plan Subsection'][TSET_IDX]


IDX = indirect_wrongs[0][0]


# In[129]:


indirect_wrong_text = []
for x in indirect_wrongs[0]:
    indirect_wrong_text.append(dataset['Assessment'][x] + "<SEP>" + dataset['Plan Subsection'][x])



import re 
def drop_spacy_annotations(texts):
    outs = []
    for text in texts:
        outs.append(re.sub(r'<\/?secondary_problem>|<\/?problem>|<\/?primary_problem>|<\/?primary_symptom>|<\/?primary_sign>|<\/?complication_related_to_problem>|<\/?event_related_to_problem>|<\/?organ_failure_related_to_problem>', 
                           '', text))
    return outs


# In[139]:


indirect_wrong_text_dropped = drop_spacy_annotations(indirect_wrong_text)


# IDX2 = list(zip(range(len(indirect_wrongs[0])), indirect_wrongs[0]))#[(0,0),(1,7),(2,8),]# (1,10), (2,16), (3, 32), (4, 34)]
# IDX1 = 1
# for x, y in IDX2:
#     print(indirect_wrong_text[x]), print("\n"), print(indirect_wrong_text_dropped[x]), print(f"Predicted: {id2label[preds[y]]}"), print(f"True: {id2label[1]}")
#     print("\n\n-----------------------------------------------------\n\n")

fig, ax = plt.subplots(figsize=(12,12))
ax.tick_params(axis='both', which='major', labelsize=13)
ax.tick_params(axis='both', which='minor', labelsize=13)
fig.suptitle('Validation Set Confusion Matrix', fontsize=20, weight="bold")
plt.xlabel('Predicted Label', fontsize=16, weight="bold")
plt.ylabel('True Label', fontsize=16, weight="bold")
plt.tight_layout()
ConfusionMatrixDisplay.from_predictions(dataset['label'].numpy(), preds, ax=ax, display_labels=['Direct', 'Indirect', 'Neither', 'Not Relevant'])
plt.savefig("conf.png",dpi=300, bbox_inches = "tight")