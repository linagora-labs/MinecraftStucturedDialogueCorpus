
We annotated all except for 7 dialogues in the original Minecraft Discourse Corpus. 
We preserve the original splits in these files:

TRAIN_307
VAL_100
TEST_133

However, when we were testing the parser, we took a subset of the original test set
to use as our development set. We also combine train + val and use a smaller validation set 
when training the parser. This cut of the data is found in these files:

TRAIN+VAL_407   (combine train and val from original split) 
DEV_32		(32 games taken at random from original test split)	
TEST_101        (Remaining test is used at test)


NB: The corpus files for BertLine have compressed builder actions. Versions of these files with full builder actions
are in the reformat directory, and can be regenerated using the action_format.py script.

Note: A few builder actions in the original corpus fell outside the grid boundaries, but are included anyway. They are:

C145-B35-A15 in DEV_32 and TEST_133 --> Builder: 1rh3t

C48-B1-A44 in VAL_100 and TRAIN+VAL_407 -->
Builder: 1oh7w 1oh6w 1oh6s
Builder: 0oh6s 0oh7w 0oh6w
