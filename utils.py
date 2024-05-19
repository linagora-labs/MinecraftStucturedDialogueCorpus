import json
import re
import numpy as np
from collections import Counter
import torch

def load_data(filename, map_relations):

    """
    load json data
    remove backwards relations and invalid chars
    return data stats
    """
    print("Loading data:", filename)
    with open(filename, "r") as f_in:
        inp = f_in.read()
        data = json.loads(inp)
        cnt_edus = 0
        cnt_relations = 0
        cnt_relations_backward = 0
        cnt_multi_parents = 0

        for dialog in data:
            last_speaker = None
            turn = 0

            for edu in dialog["edus"]:

                cnt_edus += 1
                text = edu['text']

                invalid_chars = ["/", "\*", "^", ">", "<", "\$", "\|", "=", "@"]
                for ch in invalid_chars:
                    text = re.sub(ch, "", text)

                edu['text'] = text

                if edu["speaker"] != last_speaker:
                    last_speaker = edu["speaker"]
                    turn += 1
                edu["turn"] = turn

            forwards_only = []
            for relation in dialog["relations"]:
                cnt_relations += 1
                if relation['x'] > relation['y']:
                    cnt_relations_backward += 1
                else:
                    relation["type"] = map_relations[relation["type"]]
                    forwards_only.append(relation)

            multi_parent = Counter([elem['y'] for elem in forwards_only])
            cnt = len([i for i in multi_parent.items() if i[1] > 1])
            cnt_multi_parents += cnt

            dialog['relations'] = forwards_only

    print("%d dialogs, %d edus, %d relations, %d backward relations" % (len(data), cnt_edus, cnt_relations, cnt_relations_backward))
    print("%d edus have multiple parents" % cnt_multi_parents)

    return data

def multi_delete(list_, indexes):
    indexes = sorted(list(indexes), reverse=True)
    for index in indexes:
        del list_[index]
    return list_

def input_format_linear(data, max_distance, passno = 1):
    """
     Takes loaded data json and a max distance int
    Returns candidate pairs and labels within max distance (if not test data)
    returns raw text for positions
    **does not flatten the data

    raw text : a list of strings, one string per turn. ['Build: Mission has...', 'Archi: Hello...', ]
    NB: raw is a list of len == #games
    ***
    input text is a list of lists of candidate pairs. 
    Len == sum(len(games) in raw). sum(len(r) for r in raw_train) - len(raw_train)
    NB: it is a list of list of lists.
    
    ***
    TEMP holds all the information for each edu, including game index, x, y indices, 0 for attach or no, 
    -1 for rel type. 0 and -1 are replaced by attach or no and rel type code. Once these have been replaced, 
    they are the labels.
    
    """
    # build the samples and targets :
    input_text, labels, raw = [], [], []
    for i in range(len(data)):

        raw_text = [j["speaker"][:5] + ": " + j["text"] for j in data[i]["edus"] ]
        raw += [raw_text]
        
        if passno == 2:
            #add 4 extra slots for edu information
            # temp = [[ [i, cand, y, 0, -1, 0, 0, 0, 0 ] for cand in range(y)] for y in range(1, len(data[i]["edus"]))]
            temp = [[ [i, cand, y, 0, -1, 0, 0] for cand in range(y)] for y in range(1, len(data[i]["edus"]))]
        else:
            temp = [[ [i, cand, y, 0, -1 ] for cand in range(y)] for y in range(1, len(data[i]["edus"]))]
    
        for rel in data[i]['relations']:
            #first index is the first list index, eg. 1 for y index 2
            #second index is the x index for the list in the first list
            temp[rel['y']-1][rel['x']][3] = 1 
            temp[rel['y']-1][rel['x']][4] = rel['type'] 
        
        if passno == 2:
            for yk in range(1, len(data[i]['edus'])):
                for xk in range(yk):
                    # temp[yk-1][xk][5] = data[i]['edus'][xk]['type']
                    # temp[yk-1][xk][6] = data[i]['edus'][yk]['type']
                    # temp[yk-1][xk][7] = data[i]['edus'][xk]['res']
                    # temp[yk-1][xk][8] = data[i]['edus'][yk]['res']
                    temp[yk-1][xk][5] = data[i]['edus'][xk]['res']
                    temp[yk-1][xk][6] = data[i]['edus'][yk]['res']
                    # temp[yk-1][xk][7] = int(bool(data[i]['edus'][xk]['turn_ind'] <=4))
                    # temp[yk-1][xk][8] = int(bool(data[i]['edus'][yk]['turn_ind'] <=4))
                
        labels += temp
        input_text += [[[raw_text[k-cand],raw_text[k]] for cand in range(k,0,-1)] for k in range(1,len(raw_text))]

    # delete elements with distance > max_distance
    labels = [temp[-max_distance:] for temp in labels]
    input_text = [temp[-max_distance:] for temp in input_text]
    
    #flattend list of lists of candidates into a list of candidates, idem for relations
    flat_input_text, flat_labels = [], []
    for candidate in input_text:
        flat_input_text += candidate
    for lab in labels:
        flat_labels += lab

    
    return flat_input_text, flat_labels, raw

def input_format_linear_old(data, max_distance, passno = 1, test=False):
    """
     Takes loaded data json and a max distance int
    Returns candidate pairs and labels within max distance (if not test data)
    returns raw text for positions
    **does not flatten the data

    raw text : a list of strings, one string per turn. ['Build: Mission has...', 'Archi: Hello...', ]
    NB: raw is a list of len == #games
    ***
    input text is a list of lists of candidate pairs. 
    Len == sum(len(games) in raw). sum(len(r) for r in raw_train) - len(raw_train)
    NB: it is a list of list of lists.
    
    ***
    TEMP holds all the information for each edu, including game index, x, y indices, 0 for attach or no, 
    -1 for rel type. 0 and -1 are replaced by attach or no and rel type code. Once these have been replaced, 
    they are the labels.
    
    """
    # build the samples and targets :
    input_text, labels, raw = [], [], []
    for i in range(len(data)):

        raw_text = [j["speaker"][:5] + ": " + j["text"] for j in data[i]["edus"] ]
        raw += [raw_text]
        
        if passno == 2:
            #add 4 extra slots for edu information
            # temp = [[ [i, cand, y, 0, -1, 0, 0, 0, 0 ] for cand in range(y)] for y in range(1, len(data[i]["edus"]))]
            temp = [[ [i, cand, y, 0, -1, 0, 0] for cand in range(y)] for y in range(1, len(data[i]["edus"]))]
        else:
            temp = [[ [i, cand, y, 0, -1 ] for cand in range(y)] for y in range(1, len(data[i]["edus"]))]
    
        for rel in data[i]['relations']:
            #first index is the first list index, eg. 1 for y index 2
            #second index is the x index for the list in the first list
            temp[rel['y']-1][rel['x']][3] = 1 
            temp[rel['y']-1][rel['x']][4] = rel['type'] 
        
        if passno == 2:
            for yk in range(1, len(data[i]['edus'])):
                for xk in range(yk):
                    # temp[yk-1][xk][5] = data[i]['edus'][xk]['type']
                    # temp[yk-1][xk][6] = data[i]['edus'][yk]['type']
                    # temp[yk-1][xk][7] = data[i]['edus'][xk]['res']
                    # temp[yk-1][xk][8] = data[i]['edus'][yk]['res']
                    temp[yk-1][xk][5] = data[i]['edus'][xk]['res']
                    temp[yk-1][xk][6] = data[i]['edus'][yk]['res']
                    # temp[yk-1][xk][7] = int(bool(data[i]['edus'][xk]['turn_ind'] <=4))
                    # temp[yk-1][xk][8] = int(bool(data[i]['edus'][yk]['turn_ind'] <=4))
                
        labels += temp
        input_text += [[[raw_text[k-cand],raw_text[k]] for cand in range(k,0,-1)] for k in range(1,len(raw_text))]

    if test:
        return input_text, labels, raw
    else:
        # delete elements with distance > max_distance
        labels = [temp[-max_distance:] for temp in labels]
        input_text = [temp[-max_distance:] for temp in input_text]
    
    return input_text, labels, raw

def tokenize(input, tokenizer, device): 
    """
    this is passed the tokenizer and device instantiation from the notebook
    """

    batch_tokenized = tokenizer(input, return_tensors="pt", padding=True, truncation=True, add_special_tokens=True) # get tokens id for each token (word) in the dialog
    input_ids = batch_tokenized["input_ids"].to(device) # list of token ids of dialogs in batch 
    attention_masks = batch_tokenized["attention_mask"].to(device) # cuda
    token_type_ids = batch_tokenized["token_type_ids"].to(device)

    return input_ids, attention_masks, token_type_ids

def encode_data_linear(tokenizer, device, input_text, labels, raw):
    """
    need to instantiate tokenizer before running this in notebook
    does not return tokens
    """
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
    
    #put data in correct format for linear
    input_ids, attention_masks, token_type_ids, position_ids = [], [], [], []

    for i in range(len(input_text)):
        input_ids_, attention_masks_, token_type_ids_ = tokenize(input_text[i], tokenizer, device)
        position_ids_ = []
        for e, label in enumerate(labels[i]) : 
            position_ids_vector_ = [0]
            position_ids_vector_ += positions[label[0]][label[1]]
            position_ids_vector_ += positions[label[0]][label[2]]
            position_ids_vector_ = [t-position_ids_vector_[1]+1 if t != 0  else 0 for t in position_ids_vector_]
            position_ids_vector_ += [0 for i in range(len(input_ids_[e])-len(position_ids_vector_))]
            if len(position_ids_vector_) > 512:
                position_ids_ += [position_ids_vector_[:512]]
            else:
                position_ids_ += [position_ids_vector_]
        position_ids_ = torch.tensor(position_ids_)
        input_ids += [input_ids_]
        attention_masks += [attention_masks_]
        token_type_ids += [token_type_ids_]
        position_ids += [position_ids_] 
        
    return input_ids, attention_masks, token_type_ids, position_ids

def input_format(data, max_distance, relations=False, attach_preds=None):
    """
    Takes loaded data json and a max distance int
    Returns candidate pairs and labels within max distance
    labels = [dialog index, x index, y index, 1/0*, label index]
    *only if relations == False
    Also returns raw -- a list of lists of dialogue text 
    to be used in position calculation
    NB: if relations == True, then return only the candidates with the relations
    """
    # build the samples and targets :
    input_text, input_text_, labels_, labels_complete, raw = [], [], [], [], []
    for i in range(len(data)):
        #print("now working on", i)

        raw_text = [j["speaker"][:5] + ": " + j["text"] for j in data[i]["edus"] ]
        raw += [raw_text]
        
        if relations:

            if attach_preds is not None:

                labels_ = [[i, elem[0], elem[1], -1] for elem in attach_preds[i]]
            else:
        
                labels_ = [[i, elem['x'], elem['y'], elem['type']] for elem in data[i]['relations']]

            input_text += [[raw_text[labels_[j][1]],raw_text[labels_[j][2]]] for j in range(len(labels_))]

            labels_complete += labels_

        else:
     
            temp = [[ [i, cand, y, 0, -1 ] for cand in range(y)] for y in range(1, len(data[i]["edus"]))]

            for en, rel in enumerate(data[i]['relations']):
                try:
                    temp[rel['y']-1][rel['x']][3] = 1 
                    temp[rel['y']-1][rel['x']][4] = rel['type'] 
                except IndexError as e:
                    print('{} on data index {} on turn {}'.format(e, i, en))
            
            labels_ += temp
            input_text_ += [[[raw_text[k-cand],raw_text[k]] for cand in range(k,0,-1)] for k in range(1,len(raw_text))]
    
    if relations:
        long_indices = [i for i in range(len(labels_complete)) if labels_complete[i][2]-labels_complete[i][1]>max_distance]
        input_text = multi_delete(input_text, long_indices)
        labels_complete = multi_delete(labels_complete, long_indices)

    else:
        #flattened list of lists of candidates into a list of candidates, idem for relations
        for candidate in input_text_ :
            input_text += candidate
        for lab in labels_:
            labels_complete += lab
        #remove candidates over a specified max distance
        long_indices = [i for i in range(len(labels_complete)) if labels_complete[i][2]-labels_complete[i][1]>max_distance]
        input_text = multi_delete(input_text, long_indices)
        labels_complete = multi_delete(labels_complete, long_indices)

    if relations:
        print('relation types only...')
        print('{} relations/candidates'.format(len(labels_complete)))
    else:
        num_rels = len([r for r in labels_complete if r[3] == 1])
        print('{} relations'.format(num_rels))
        print('{} candidates'.format(len(labels_complete)))
        print('{} non attached'.format(len(labels_complete) - num_rels))

    return input_text, labels_complete, raw

def position_ids_compute(tokenizer, input_ids, raw, labels):  # not finished
    ''' Compute position_ids vector for bert component'''
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
    
    return position_ids
#position_ids = torch.tensor(position_ids) !! need to tensorize the output!

def id_delete(tensor, indices):
    mask = torch.ones(len(tensor), dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]

def undersample(n_keep, labels):
    #find the indicies of unattached labels 3
    arglist = [i for i in range(len(labels)) if labels[i][3] == 0]
    indices = sorted(np.random.choice(len(arglist),len(arglist)-n_keep,replace=False))
    arglist = list(np.array(arglist)[indices])
   
    return arglist

# def undersample_multi(n_keep, labels):
#     #find the indicies of unattached labels 3
#     arglist = [i for i in range(len(labels)) if labels[i][3] == -1]
#     indices = sorted(np.random.choice(len(arglist),len(arglist)-n_keep,replace=False))
#     arglist = list(np.array(arglist)[indices])
   
#     return arglist

def flatten(listoflists):
    flat_version = []
    for l in listoflists:
        flat_version += l
    return flat_version

def get_batch_ids(len_data, batch_size):
    """
    returns a list of lists of indices, 16 indicies per list
    to be used in creating bert batches
    """
    indices = [i for i in range(len_data)]
    batches = []
    for i in range(len_data // batch_size + bool(len_data) % batch_size):
        batches.append(indices[i * batch_size:(i + 1) * batch_size])
    return batches

