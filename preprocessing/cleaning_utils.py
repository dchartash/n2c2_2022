import re 
import string
from spacy.tokens import Doc

def extract_text_segments(example, mimic_note_df=None):
    '''
    Given an example from the hf.Dataset class, we extract the text component from the mimic_note_df and return it as a new column.
    Meant to be used with hf.Dataset.map()
    
    NOT BEING USED, since we get the text without having to do span extraction.
    '''
    example['Assessment'] = mimic_note_df[mimic_note_df['ROW_ID'] == example["ROW ID"]]['TEXT'].values[0][example["Assessment Begin"]:example["Assessment End"]]
    example['PlanSubsection'] = mimic_note_df[mimic_note_df['ROW_ID'] == example["ROW ID"]]['TEXT'].values[0][example["PlanSubsection Begin"]:example["PlanSubsection End"]]
    
    return example


def tokenize_function(examples, tokenizer=None):
    return tokenizer(examples['Assessment'], examples['Plan Subsection'],
                                      truncation="only_second",
                                      # truncation="longest_first",                     
                                      max_length=512,
                                      verbose=True)
                    

def remove_mimic_deid(example):
    
    example['Assessment'] = re.sub(r"\[\*\*.*?\*\*\]", "", example['Assessment'])
    example['Plan Subsection'] = re.sub(r"\[\*\*.*?\*\*\]", "", example['Plan Subsection'])    
    
    return example    
    
def add_ner_assessment(example, nlp=None):
    '''
    Takes a trained spacy model and tags with the NER tags we trained on for assessment
    '''
    tagged = ""
    orig = example['Assessment']
    doc = nlp(example['Assessment'])
    
    index = 0
    for ent in doc.ents:
        tagged += orig[index: ent.start_char] + "<"+ ent.label_ + ">" + orig[ent.start_char:ent.end_char] + "</" + ent.label_ + ">"
        index = ent.end_char
    tagged += orig[index:]

    example['Assessment'] = tagged
    return example


def add_ner_plan(example, nlp=None):
    '''
    Takes a trained spacy model and tags with the NER tags we trained on for assessment
    '''
    tagged = ""
    orig = example['Plan Subsection']
    doc = nlp(example['Plan Subsection'])
    
    index = 0
    for ent in doc.ents:
        tagged += orig[index: ent.start_char] + "<"+ ent.label_ + ">" + orig[ent.start_char:ent.end_char] + "</" + ent.label_ + ">"
        index = ent.end_char
    tagged += orig[index:]

    example['Plan Subsection'] = tagged
    return example


def add_ner_assessment_end(example, nlp=None):
    '''
    Takes a trained spacy model and tags with the NER tags we trained on for assessment
    '''
    tagged = ""
    orig = example['Assessment']
    doc = nlp(example['Assessment'])
    
    for ent in doc.ents:
        tagged += "</" + ent.label_ + ">"

    example['Assessment'] = orig + " " + tagged
    return example


def add_ner_plan_end(example, nlp=None):
    '''
    Takes a trained spacy model and tags with the NER tags we trained on for assessment
    '''
    tagged = ""
    orig = example['Plan Subsection']
    doc = nlp(example['Plan Subsection'])
    
    index = 0
    for ent in doc.ents:
        tagged += "</" + ent.label_ + ">"

    example['Plan Subsection'] = orig + " " + tagged
    return example

def add_ner_so(example, nlp=None):
    '''
    Takes a trained spacy model and tags with the NER tags we trained on for S&O
    '''
    tagged = ""
    orig = example['S']
    doc = nlp(example['S'])
    
    index = 0
    for ent in doc.ents:
        tagged += orig[index: ent.start_char] + "<"+ ent.label_ + ">" + orig[ent.start_char:ent.end_char] + "</" + ent.label_ + ">"
        index = ent.end_char
    tagged += orig[index:]

    example['S'] = tagged
    
    tagged = ""
    orig = example['O']
    doc = nlp(example['O'])
    
    index = 0
    for ent in doc.ents:
        tagged += orig[index: ent.start_char] + "<"+ ent.label_ + ">" + orig[ent.start_char:ent.end_char] + "</" + ent.label_ + ">"
        index = ent.end_char
    tagged += orig[index:]

    example['O'] = tagged
    
    return example

    
def split_leading_symptom_list(example):
    '''Mapping function to split text with a leading symptom list, if it exists, and return the rest of the assessment
    '''
    # get all the text that is all caps at the beginning, (ignoring capital case)
    pattern = re.compile(r"^([A-Z\W]{2,})")
    pattern2 = re.compile(r"[A-Z]")
    example['Symptom List'] = None
    splits = pattern.split(example['Assessment'])
    splits = [split for split in splits if split != ""]
    # if the pattern isn't found, we do nothing
    if len(splits) == 1:
        example['Symptom List'] = None
    else:
        # because of our pattern, we have to move the last letter from the first split to the next split (if it exists)
        if pattern2.match(splits[0][-1]):
            # print(example['ROW ID'])
            last_letter = splits[0][-1]
            splits[0] = splits[0][:-1]
            splits[1] = last_letter + splits[1]
            example['Symptom List'] = splits[0]
            example['Assessment'] = "".join(splits[1:])

    return example
    
    
def expand_abbreviations(example, spacy_pip=None, abbv_map=None, unq_sfs=None):
    '''
    Takes in a spacy pipeline as spacy_pip, abbv_map which is the abbv dataframe, and a unique short form set of abbvs
    '''
    assessment_doc = spacy_pip.tokenizer(example['Assessment'])
    plan_doc = spacy_pip.tokenizer(example['Plan Subsection'])
        
    # we've sorted the abbv_map with the most likely abbrevation sets so we take the first one in the list and assume it's right

    words = []
    spaces = []
    for token in assessment_doc:
        if token.text in unq_sfs:
            abbv_doc = spacy_pip.tokenizer(abbv_map[abbv_map["SF"] == token.text].iloc[0].squeeze().at["LF"])
            for tok in abbv_doc:
                words.append(tok.text)
                spaces.append(True)
        else:
            words.append(token.text)
            spaces.append(True if token.whitespace_ == " " else False)
    
    example['Assessment'] = Doc(vocab=spacy_pip.vocab, words=words, spaces=spaces).text
    
    words = []
    spaces = []
    for token in plan_doc:
        if token.text in unq_sfs:
            abbv_doc = spacy_pip.tokenizer(abbv_map[abbv_map["SF"] == token.text].iloc[0].squeeze().at["LF"])
            for tok in abbv_doc:
                words.append(tok.text)
                spaces.append(True)
        else:
            words.append(token.text)
            spaces.append(True if token.whitespace_ == " " else False)
    
    example['Plan Subsection'] = Doc(vocab=spacy_pip.vocab, words=words, spaces=spaces).text
    
    return example


def remove_MIMIC_deid(example):
    pat = re.compile("\[\*\*.*?\*\*\]")
    example['Assessment'] = pat.sub("PLACEHOLDER", example['Assessment'])
    
    example['Plan Subsection'] = pat.sub("", example['Plan Subsection'])
    return example    

def add_SO_sections(example):
    
    prepend = ""
    if example['S']:
        x = example['S'] + " "
        prepend += x.strip()
    if example['O']:
        x = example['O'] + " "
        prepend += x.strip()

    # example['Assessment'] = prepend + example['Assessment']
    # example['Assessment'] = example['Assessment'].strip()
    example['Plan Subsection'] = example['Plan Subsection'] + "</s>" + prepend
    example['Plan Subsection'] = example['Plan Subsection'].strip()
    
    # example['Plan Subsection'] = pat.sub("", example['Plan Subsection'])
    return example    

