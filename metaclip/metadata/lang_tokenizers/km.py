# Copyright (c) Meta Platforms, Inc. and affiliates


# Source: https://github.com/phylypo/segmentation-crf-khmer/

print("Loading km tokenizer...", end=' ')

import pickle
from xopen import xopen
from typing import List

# add the path of my_utils folder
import os
# Get the directory of the current script
dir_path = os.path.dirname(os.path.realpath(__file__))
# Construct the path to the model file
model_file_path = os.path.join(dir_path, 'km.sav.gz')

# load the model
crf = pickle.load(xopen(model_file_path, 'rb'))

#@title Segment into KCCs 

# list of constants needed for KCC and feature generation
# consonant and independent vowels
KHCONST = set(u'កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវឝឞសហឡអឣឤឥឦឧឨឩឪឫឬឭឮឯឰឱឲឳ')
KHVOWEL = set(u'឴឵ាិីឹឺុូួើឿៀេែៃោៅ\u17c6\u17c7\u17c8')
# subscript, diacritics
KHSUB = set(u'្')
KHDIAC = set(u"\u17c9\u17ca\u17cb\u17cc\u17cd\u17ce\u17cf\u17d0") #MUUSIKATOAN, TRIISAP, BANTOC,ROBAT,
KHSYM = set('៕។៛ៗ៚៙៘,.? ') # add space
KHNUMBER = set(u'០១២៣៤៥៦៧៨៩0123456789') # remove 0123456789
# lunar date:  U+19E0 to U+19FF ᧠...᧿
KHLUNAR = set('᧠᧡᧢᧣᧤᧥᧦᧧᧨᧩᧪᧫᧬᧭᧮᧯᧰᧱᧲᧳᧴᧵᧶᧷᧸᧹᧺᧻᧼᧽᧾᧿')

def is_khmer_char(ch):
  if (ch >= '\u1780') and (ch <= '\u17ff'): return True
  if ch in KHSYM: return True
  if ch in KHLUNAR: return True
  return False

def is_start_of_kcc(ch):
  if is_khmer_char(ch):
    if ch in KHCONST: return True
    if ch in KHSYM: return True
    if ch in KHNUMBER: return True
    if ch in KHLUNAR: return True
    return False
  return True

# kcc base - must surround space with \u200b using cleanupstr()
def seg_kcc(str_sentence):
    segs = []
    cur = ""
    sentence = str_sentence
    #for phr in str_sentence.split(): #no longer split by space, use 200b
    #    print("phr: '", phr,"'")
    for word in sentence.split('\u200b'):
      #print("PHR:[%s] len:%d" %(phr, len(phr)))
      for i,c in enumerate(word):
          #print(i," c:", c)
          cur += c
          nextchar = word[i+1] if (i+1 < len(word)) else ""
          
          # cluster non-khmer chars together
          if not is_khmer_char(c) and nextchar != " " and nextchar != "" and not is_khmer_char(nextchar): 
            continue
          # cluster number together
          if c in KHNUMBER and nextchar in KHNUMBER: 
            continue
            
          # cluster non-khmer together
          # non-khmer character has no cluster
          if not is_khmer_char(c) or nextchar==" " or nextchar=="":
              segs.append(cur)
              cur=""
          elif is_start_of_kcc(nextchar) and not (c in KHSUB):
              segs.append(cur)
              cur="" 
        # add space back after split
        #segs.append(" ")   
    return segs # [:-1] # trim last space

EN = set(u'abcdefghijklmnopqrstuvwxyz0123456789')

# E=English, C=Consonant, W=wowel, N=number, O=Other, S=subcript, D=Diacritic, NS=no_space(same E)
# roll up to: NS, C, W, S, D
NS = 'NS'
def get_type(chr):
  if chr.lower() in EN: return NS
  if chr in KHCONST: return "C"
  if chr in KHVOWEL: return "W"
  if chr in KHNUMBER: return NS
  if chr in KHSUB: return "S"
  if chr in KHDIAC: return "D"
  return NS

# non-khmer character that we should not separate like number
# multiple characters are false
def is_no_space(k):
  if get_type(k[0])==NS: return True
  return False

def kcc_type(k):
  if len(k)==1: return get_type(k)
  else: return "K" + str(len(k))
def kcc_to_features(kccs, i):
    maxi = len(kccs)
    kcc = kccs[i]

    features = {
        'kcc': kcc,
        't': kcc_type(kcc),
        'ns': is_no_space(kcc)
    }
    if i >= 1:
        features.update({
            'kcc[-1]'  : kccs[i-1],
            'kcc[-1]t' : kcc_type(kccs[i-1]),
            'kcc[-1:0]': kccs[i-1] + kccs[i],
            'ns-1' : is_no_space(kccs[i-1])
        })
    else:
        features['BOS'] = True

    if i >= 2:
        features.update({
            'kcc[-2]'   : kccs[i-2],
            'kcc[-2]t'  : kcc_type(kccs[i-2]),
            'kcc[-2:-1]': kccs[i-2] + kccs[i-1],
            'kcc[-2:0]' : kccs[i-2] + kccs[i-1] + kccs[i],
        })
    if i >= 3:
        features.update({
            'kcc[-3]'   : kccs[i-3],
            'kcc[-3]t'  : kcc_type(kccs[i-3]),
            'kcc[-3:0]' : kccs[i-3] + kccs[i-2] + kccs[i-1] + kccs[i],
            'kcc[-3:-1]': kccs[i-3] + kccs[i-2] + kccs[i-1],
            'kcc[-3:-2]': kccs[i-3] + kccs[i-2],
        })

    if i < maxi-1:
        features.update({
            'kcc[+1]'  : kccs[i+1],
            'kcc[+1]t'  : kcc_type(kccs[i+1]),
            'kcc[+1:0]': kccs[i] + kccs[i+1],
            'ns+1' : is_no_space(kccs[i+1])

        })
    else:
        features['EOS'] = True

    if i < maxi-2:
        features.update({
            'kcc[+2]'   : kccs[i+2],
            'kcc[+2]t'   : kcc_type(kccs[i+2]),
            'kcc[+1:+2]': kccs[i+1] + kccs[i+2],
            'kcc[0:+2]' : kccs[i+0] + kccs[i+1] + kccs[i+2],
            'ns+2' : is_no_space(kccs[i+2])
        })
    if i < maxi-3:
        features.update({
            'kcc[+3]'   : kccs[i+3],
            'kcc[+3]t'   : kcc_type(kccs[i+3]),
            'kcc[+2:+3]': kccs[i+2] + kccs[i+3],
            'kcc[+1:+3]': kccs[i+1] + kccs[i+2] + kccs[i+3],
            'kcc[0:+3]' : kccs[i+0] + kccs[i+1] + kccs[i+2] + kccs[i+3],
        })

    return features
def create_kcc_features(kccs):
    return [kcc_to_features(kccs, i) for i in range(len(kccs))]
def segment_text(str, spacer=" "):
  complete = ""
  for sen in str.split('\n'):
    if sen.strip() == "": continue
    sen = sen.replace(u'\u200b','')
    kccs = seg_kcc(sen)
    features=create_kcc_features(kccs)
    # predicts take list of sentences features
    prediction = crf.predict([features])

    #print("-len kccs:", len(kccs), " data 13:", kccs[:13])
    #print("-len feature:", len(features), " data 3:", features[:3])
    #print("-len prediction:", len(prediction), "data[0]:", prediction[0])

    for i, p in enumerate(prediction[0]):
        if p == "1":
            complete += spacer + kccs[i]
        else:
            complete += kccs[i]
    complete += "\n"
  complete = complete.replace(spacer+" ", " ").replace(" "+spacer, " ") # no 200b before or after space
  return complete[:-1]

print('km tokenizer ready!')
def tokenize(
        texts: List[str]
) -> List[List[str]]:
    return [
        segment_text(text).split() for text in texts
    ]