
# Minecraft Structured Dialogue Corpus 

This repo contains the annotated corpus files and the code for the BertLine discourse parser. 
The data and the model are described in the 2024 LREC paper (forthcoming)

Note:  Some minor corrections have been made to the data since LREC publication. The updated stats are summarized below. 

# Data
### Corpus Stats

|                  | Train+Val | Test | Total |
|----------------|-------|--------|------|
|No. Dialogues | 407| 133| 540|
|No. EDUs | 17135 | 5402 | 22537 |
|No. EDUs (squished) |4687|1473|6160| 
|No. Relation instances |26279|8250|34529|
|No. Multi-parent DUs |4789|1476|6265|


#### Breakdown by relation type and endpoint type*. 
*'Lin' is EDU and 'NL' is nonlinguistic DU, or EEU.
#### Forward Relations                                                
                                     
|                       | Lin-Lin  |Lin-NL |NL-Lin  |NL-NL  |Total|
|-----------------------|----------|-------|--------|-------|-----|
|Elaboration            |    4076  |     0 |      0 |     0 | 4076|
|Acknowledgement        |    1814  |     0 |    2736|      0| 4550|
|Continuation           |    2029  |     0 |       0|      0| 2029|
|Contrast               |    398   |     0 |       0|      0|  398|
|Correction             |    230   |     0 |    964 |    968| 2162|
|Result                 |    1944  |  6015 |    2369|      0|10418|
|Comment                |    1512  |     0 |    164 |      0| 1676|
|Question-answer_pair   |    1933  |     0 |      0 |      0| 1933|
|Narration              |     4455 |     0 |    0   |      0|4455 |
|Clarification_question |    960   |     0 |    0   |      0| 960 |
|Confirmation_question  |    43    |     0 |     956|      0|  999|
|Q-Elab                 |    229   |     0 |    0   |      0|  229|
|Explanation            |    108   |     0 |    0   |      0|  108|
|Alternation            |    173   |     0 |    0   |      0|  173|
|Conditional            |    67    |     0 |    0   |      0|   67|
|Sequence               |    0     |     38|    0   |      0|   38|

#### Backwards Relations 
|                       | Lin-Lin  |Lin-NL |NL-Lin  |NL-NL  |Total|
|-----------------------|----------|-------|--------|-------|-----| 
|Comment                |    242   |     0 |    0   |      0| 242 |
|Conditional            |    16    |     0 |    0   |      0| 16  |


# Parser 

We use the BertLine parser as originally described in [Bennis et al, 2023](https://aclanthology.org/2023.eacl-main.247.pdf). 

#### To reproduce our results, run the notebooks in this order:
 1.  bert_finetune.ipynb
 2.  bert_linear.ipynb
 3.  bert_multitaks.ipynb
 4.  bert_multitask_test.ipynb
 
