def extract_text_segments(example, mimic_note_df=None):
    '''
    Given an example from the hf.Dataset class, we extract the text component from the mimic_note_df and return it as a new column.
    Meant to be used with hf.Dataset.map()
    '''
    example['Assessment'] = mimic_note_df[mimic_note_df['ROW_ID'] == example["ROW ID"]]['TEXT'].values[0][example["Assessment Begin"]:example["Assessment End"]]
    example['PlanSubsection'] = mimic_note_df[mimic_note_df['ROW_ID'] == example["ROW ID"]]['TEXT'].values[0][example["PlanSubsection Begin"]:example["PlanSubsection End"]]
    
    return example



def tokenize_function(examples):
    return bio_clinicalbert_tokenizer(examples['Assessment'], examples['Plan Subsection'],
                                      truncation="longest_first",
                                      max_length=512,
                                      verbose=True)
                    
