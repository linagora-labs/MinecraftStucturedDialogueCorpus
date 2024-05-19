old_map_relations = {'Comment': 0, 'Contrast': 1, 'Correction': 2, 'Question-answer_pair': 3, 'QAP': 3, 'Parallel': 4, 'Acknowledgement': 5,
            'Elaboration': 6, 'Clarification_question': 7, 'Conditional': 8, 'Continuation': 9, 'Result': 10, 'Explanation': 11,
            'Q-Elab': 12, 'Alternation': 13, 'Narration': 14, 'Confirmation_question': 15, 'Break': 16, 'Sequence' : 17}

use_cuda = True
if use_cuda :
    device = 'cuda'
else : device = 'cpu'

vocab_refining = True

max_distance = 10

model_name = "bert-base-cased"

data_set = 'minecraft'#'stac' # or 'stac_squished'
if data_set == 'minecraft' :
    data_path = 'data/minecraft_data/'
    model_path = 'models/minecraft/'

# elif data_set == 'stac_squished' :
#     data_path = 'data/stac_squished_data/'
#     model_path = 'models/stac_squished/'

seed = 18
valid_size = 20 #90

bert_name = 'bert_finetuned'
linear_name =  'linear'

# bert parameters
n_keep = 45000 # 30000 # 20000 #50000 #10000
batch_size_bert = 32 #16
eps_bert = 1e-8
lr_bert = 1.5e-5 #2e-5 #3e-5 #5e-5  #1.5e-5
epochs_bert = 2

# linear parameters
batch_size_linear = 16 #32 #16
lr_linear = 0.0002
epochs_linear = 15
hidden_size = 770
hidden_size_1 = 2000

# multitask parameters
n_keep_multi = 30000
epoch_multitask = 3



     