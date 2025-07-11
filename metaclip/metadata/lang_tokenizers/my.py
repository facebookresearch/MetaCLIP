# Copyright (c) Meta Platforms, Inc. and affiliates


# Word Segmentation for Burmese Language using Conditional Random Field
# Author : Thura Aung
# 5 July, 2022

print('Loading my tokenizer...', end=' ')

import pycrfsuite
from typing import List

# add the path of my_utils folder
import os
# Get the directory of the current script
dir_path = os.path.dirname(os.path.realpath(__file__))
# Construct the path to the model file
model_file_path = os.path.join(dir_path, 'my.crfsuite')

#open trained model
tagger = pycrfsuite.Tagger()
tagger.open(model_file_path)

#here sentence is prepared_sentence and i is length of prepared_sentence
def create_char_features(sentence, i):
    #set initial feature set char as first char in prepared_sentence
    features = [
        'bias',
        'char=' + sentence[i][0] 
    ]
    #if i >=1 then go to previous character else append 'BOS' in features list 
    if i >= 1:
        features.extend([
            'char-1=' + sentence[i-1][0],
            'char-1:0=' + sentence[i-1][0] + sentence[i][0],
        ])
    else:
        features.append("BOS")
        
    if i >= 2:
        features.extend([
            'char-2=' + sentence[i-2][0],
            'char-2:0=' + sentence[i-2][0] + sentence[i-1][0] + sentence[i][0],
            'char-2:-1=' + sentence[i-2][0] + sentence[i-1][0],
        ])
        
    if i >= 3:
        features.extend([
            'char-3:0=' + sentence[i-3][0] + sentence[i-2][0] + sentence[i-1][0] + sentence[i][0],
            'char-3:-1=' + sentence[i-3][0] + sentence[i-2][0] + sentence[i-1][0],
        ])
    #if i+1 < len(sentence) then go to next character and set it to next character and set char to next two characters else append 'EOS' to features list
    if i + 1 < len(sentence):
        features.extend([
            'char+1=' + sentence[i+1][0],
            'char:+1=' + sentence[i][0] + sentence[i+1][0],
        ])
    else:
        features.append("EOS")
    #if first if condition satisfy then go to second and third if condition and do the same work for next characters    
    if i + 2 < len(sentence):
        features.extend([
            'char+2=' + sentence[i+2][0],
            'char:+2=' + sentence[i][0] + sentence[i+1][0] + sentence[i+2][0],
            'char+1:+2=' + sentence[i+1][0] + sentence[i+2][0],
        ])
    
    if i + 3 < len(sentence):
        features.extend([
            'char:+3=' + sentence[i][0] + sentence[i+1][0] + sentence[i+2][0]+ sentence[i+3][0],
            'char+1:+3=' + sentence[i+1][0] + sentence[i+2][0] + sentence[i+3][0],
        ])
    return features



def create_word_features(prepared_sentence):
    return [create_char_features(prepared_sentence, i) for i in range(len(prepared_sentence))]

#segment word by trained model
def segment_word(sentence):
    #remove white spaces from sentence
    sent = sentence.replace(" ", "")
    #tag sentence by trained model or create sentence features 
    prediction = tagger.tag(create_word_features(sent))
    #assign 'complete' to empty string 
    complete = ""
    #apply for loop on taged sentence
    for i, p in enumerate(prediction):
        #if label of character in sentence is 1 then brack that word from that place and add into complete
        if p == "1":
            complete += " " + sent[i]
        #if label of character in sentence is 0 then add that word as it is into complete
        else:
            complete += sent[i]
    #print(type(sent))
    return complete

print("my tokenizer ready!")

def tokenize(
        texts: List[str]
) -> List[List[str]]:
    return [
        segment_word(text).split() for text in texts
    ]
