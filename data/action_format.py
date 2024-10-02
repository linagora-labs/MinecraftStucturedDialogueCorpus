import os
import json

def is_nl(edu):
    """
    if every word in alphanumeric and has len 5
    """
    nl = 1
    words = edu.split(' ')
    # print(words)
    # print(words)
    for word in [w for w in words if w != '']:
        if not contains_number(word) or len(word) != 5:
            nl = 0
            break
    # print(nl)
    return nl

def contains_number(string):
    return any(char.isdigit() for char in string)

def decode(tok_str):
    """
    takes a list bert tokens and changes them back to coordinates.
    """
    zdict = {'a':'-5', 'e' : '-4', 'i':'-3', 'o':'-2', 'u':'-1', 'p':'0', 
             'q':'1', 'r':'2', 'x': '3', 'y':'4', 'z':'5'}
    xdict = {'b': '-5', 'c' :'-4', 'd' : '-3', 'f' : '-2', 'g' : '-1', 'h':'0', 
             'j':'1', 'k':'2', 'l':'3', 'm':'4', 'n':'5'}
    colors = {'r' :'red', 'b':'blue', 'g':'green', 'o':'orange', 'y':'yellow', 'p':'purple'}
    # action = {'0' : 'pick', '1': 'place'}
    decoded = []
    for tok in tok_str.split():
        # print(tok)
        if tok[0] == '0':
            new_string = 'pick ' +  colors[tok[1]] + ' ' + xdict[tok[2]] + ' ' + tok[3] + ' ' + zdict[tok[4]]
        else:
            new_string = 'place ' + colors[tok[1]] + ' ' +  xdict[tok[2]] + ' ' + tok[3] + ' ' + zdict[tok[4]]
        decoded.append(new_string)
    moves_str = ', '.join(decoded)
    return moves_str

current_folder=os.getcwd()


#create a new folder
if not os.path.exists(current_folder + '/reformat/'):
    os.mkdir(current_folder + '/reformat/')
#get path of all json files
version_one = [f for f in os.listdir() if os.path.isfile(f) and f.split('.')[1] == 'json']


for v in version_one:
    data_path = current_folder + '/' + v
    save_path = current_folder + '/reformat/' + v.split('.')[0] + '_reformat.json'

    with open(data_path, 'r') as j:
        jfile = json.load(j)
        games = jfile

    new_version = []
    for game in games:
        new_game = {}
        new_game['id'] = game['id']
        edus = game['edus']
        new_edus = []
        for edu in edus:
            if edu['speaker'] == 'Architect':
                new_edus.append(edu)
            else:
                if is_nl(edu['text']):
                
                    formatted_moves = decode(edu['text'])
                    edu['text'] = formatted_moves
                    new_edus.append(edu)
                else:
                    new_edus.append(edu)
        new_game['edus'] = new_edus
        new_game['relations'] = game['relations']
        new_version.append(new_game)
                
    

    with open(save_path, 'w') as outfile:
        json.dump(new_version, outfile)

        
        
