import params, torch,datetime
from transformers import BertTokenizer
import numpy as np


def multi_delete(list_, indexes):
    indexes = sorted(list(indexes), reverse=True)
    for index in indexes:
        del list_[index]
    return list_

def tokenize(input_, tokenizer, token = False): 

    device = torch.device(params.device)
    batch_tokenized = tokenizer(input_, return_tensors="pt", padding=True, truncation=True, add_special_tokens=True) # get tokens id for each token (word) in the dialog
    input_ids = batch_tokenized["input_ids"].to(device) # list of token ids of dialogs in batch 
    attention_masks = batch_tokenized["attention_mask"].to(device)
    token_type_ids = batch_tokenized["token_type_ids"].to(device)
    tokens = []

    if token :    
        for t in batch_tokenized["input_ids"]:
            tokens += [tokenizer.convert_ids_to_tokens(t)]
    else : tokens = None

    return input_ids, attention_masks, token_type_ids, tokens

def input_format(data, relations = False, attach_preds = None, token = False):

    max_distance = params.max_distance

    # build the samples and targets :

    input_text, input_text_, labels_, labels_complete, raw = [], [], [], [], []
    for i in range(len(data)):

        raw_text = [j["speaker"] + ": " + j["text_raw"][:-1] for j in data[i]["edus"] ]
        raw += [raw_text]
        if not relations : 
            temp = [[ [i, cand, y, 0, -1 ] for cand in range(y)] for y in range(1, len(data[i]["edus"]))]

            for rel in data[i]['relations']:
                temp[rel['y']-1][rel['x']][3] = 1 
                temp[rel['y']-1][rel['x']][4] = rel['type'] 
            
            labels_ += temp
            input_text_ += [[[raw_text[k-cand],raw_text[k]] for cand in range(k,0,-1)] for k in range(1,len(raw_text))]
            
        else :
            if attach_preds is not None :
                labels_ = [[i, elem[0], elem[1], -1] for elem in attach_preds[i]]
            else : 
                labels_ = [[i, elem['x'], elem['y'], elem['type']] for elem in data[i]['relations']]

            input_text += [[raw_text[labels_[j][1]],raw_text[labels_[j][2]]] for j in range(len(labels_))]

            labels_complete += labels_
    if not relations :
        for candidate in input_text_ :
            input_text += candidate
        for lab in labels_:
            labels_complete += lab
        long_indices = [i for i in range(len(labels_complete)) if labels_complete[i][2]-labels_complete[i][1]>max_distance]
        input_text = multi_delete(input_text, long_indices)
        labels_complete = multi_delete(labels_complete, long_indices)
    tokenizer = BertTokenizer.from_pretrained(params.model_name, use_fast=True)
    input_ids, attention_masks, token_type_ids, tokens = tokenize(input_text, tokenizer, token=token)
    labels = [label[3] for label in list(labels_complete)]
    labels = torch.tensor(labels)
    labels_complete = torch.tensor(labels_complete)
    return input_ids, attention_masks, token_type_ids, tokens, labels, labels_complete, raw

def position_ids_compute(input_ids, raw, labels):  # not finished
    ''' Compute position_ids vector for bert component'''
    tokenizer = BertTokenizer.from_pretrained(params.model_name, use_fast=True)
    ids = [tokenizer(raw[i], return_tensors="pt", padding=True, truncation=True, add_special_tokens=True)['input_ids'] for i in range(len(raw))]
    # compute position matrix
    positions = []
    for dialog in ids :
        temporary = []
        counter = 0
        for i in range(len(dialog)):
            position_vector = [counter+j for j in range(1, len(dialog[i])) if dialog[i][j] != 0]
            counter += len(position_vector)
            temporary += [position_vector]
        positions += [temporary]

    # compute position_ids 
    position_ids = []
    for e, label in enumerate(labels) : 
        position_ids_vector = [0]
        position_ids_vector += positions[label[0]][label[1]]
        position_ids_vector += positions[label[0]][label[2]]
        position_ids_vector = [t-position_ids_vector[1]+1 if t != 0  else 0 for t in position_ids_vector]
        position_ids_vector += [0 for i in range(len(input_ids[e])-len(position_ids_vector))]
        position_ids += [position_ids_vector]
    position_ids = torch.tensor(position_ids)
    return position_ids

def id_delete(tensor, indices):
    mask = torch.ones(len(tensor), dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]

def undersample(n_keep, labels, labels_attach, input_ids, attention_masks, token_type_ids, position_ids, multi=0):
    if multi==1:
        arglist = [i for i in range(len(labels_attach)) if labels_attach[i] == -1]
    else:
        arglist = [i for i in range(len(labels_attach)) if labels_attach[i] == 0]
    indices = sorted(np.random.choice(len(arglist),len(arglist)-n_keep,replace=False))
    arglist = list(np.array(arglist)[indices])
    labels = id_delete(labels, arglist)
    labels_attach = id_delete(labels_attach, arglist)
    input_ids = id_delete(input_ids, arglist)
    attention_masks = id_delete(attention_masks, arglist)
    token_type_ids = id_delete(token_type_ids, arglist)
    position_ids = id_delete(position_ids, arglist)
    return labels, labels_attach, input_ids, attention_masks, token_type_ids, position_ids

def flat_accuracy(preds_attach, labels_attach):
    pred_flat = np.argmax(preds_attach, axis=1).flatten()
    return np.sum(pred_flat == labels_attach) / len(labels_attach)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))