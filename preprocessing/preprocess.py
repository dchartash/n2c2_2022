import pandas as pd 
import numpy as np
from preprocessing.cleaning_utils import *
from functools import partial

import transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from transformers import logging

logging.set_verbosity_warning()

def main():

    # Read MIMIC notes
    notes = pd.read_csv(mimic_dir + "NOTEEVENTS.csv")

    # create hf Dataset
    classes = ['Not Relevant', 'Neither', 'Indirect', 'Direct']

    #####
    ##### NOTE:
    #####
    # we aren't doing this step cuz they messed up the numbers
    # features = Features({
    #     'ROW ID':Value("int64"),
    #     'HADM ID':Value("int64"),
    #     'Assessment Begin':Value("int64"),
    #     'Assessment End':Value("int64"),
    #     'PlanSubsection Begin':Value("int64"),
    #     'PlanSubsection End':Value("int64"),
    #     "Relation":Value("string")
    # }) 
    # dataset = load_dataset("csv", data_files=n2c2_dir + "n2c2_sample.csv", 
    #                        features=features)
    # dataset = dataset.class_encode_column("Relation")
    # dataset = dataset.map(partial(extract_text_segments, mimic_note_df=notes))

    # instead we will use the raw text for now
    features = Features({
        'ROW ID':Value("int64"),
        'HADM ID':Value("int64"),
        'Assessment':Value("string"),
        'PlanSubsection':Value("string"),
        "Relation":Value("string")
    }) 
    dataset = load_dataset("csv", data_files=n2c2_dir + "n2c2_sample_raw.csv", 
                           features=features)
    # create encoded class labels
    dataset = dataset.class_encode_column("Relation")

    # split dataset: 80% train, 10% validation, 10% test 
    train_testvalid = dataset['train'].train_test_split(test_size=0.2)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
    split_dataset = DatasetDict({
        'train': train_testvalid['train'],
        'valid': test_valid['train'],
        'test': test_valid['test'],
    })


    split_dataset = split_dataset.map(lambda examples: tokenizer(examples['Assessment'], examples['PlanSubsection'],
                                        max_length=768, 
                                        padding=True,
                                        truncation=True,
                                        verbose=True), batched=True)

    



if __name__ == "__main__":
    # Define file paths
    mimic_dir = "/home/vs428/project/MIMIC/files/mimiciii/1.4/"
    n2c2_dir = "/home/vs428/project/n2c2/2022/N2C2-AP-Reasoning/"
    hf_modelhub_path = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(hf_modelhub_path)
    model = AutoModel.from_pretrained(hf_modelhub_path)
    main()
    