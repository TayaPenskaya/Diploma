import string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from itertools import chain
from collections import defaultdict
from nltk.corpus import wordnet as wn

from data.mpii import MPIIDataset
from gan.net import Generator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from random import randrange
from utils.vis import draw_keypoints

import argparse
import matplotlib.pyplot as plt
import json

def preprocess(text):
    stoplist = list(set(stopwords.words('english')))
    punctuation= list(string.punctuation)
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()
    
    final_text = []
    for i in tokenizer.tokenize(text):
        word = i.strip().lower() 
        if not (word in stoplist or word in punctuation or len(word) < 3):
            final_text.append(lemmatizer.lemmatize(word))
    return list(set(final_text))

def morphify(word,org_pos,target_pos):
    """ morph a word """
    synsets = wn.synsets(word, pos=org_pos)

    # Word not found
    if not synsets:
        return []

    # Get all  lemmas of the word
    lemmas = [l for s in synsets \
                   for l in s.lemmas() if s.name().split('.')[1] == org_pos]

    # Get related forms
    derivationally_related_forms = [(l, l.derivationally_related_forms()) \
                                    for l in    lemmas]

    # filter only the targeted pos
    related_lemmas = [l for drf in derivationally_related_forms \
                           for l in drf[1] if l.synset().name().split('.')[1] == target_pos]

    # Extract the words from the lemmas
    words = [l.name() for l in related_lemmas]
    len_words = len(words)

    # Build the result in the form of a list containing tuples (word, probability)
    result = [(w, float(words.count(w))/len_words) for w in set(words)]
    result.sort(key=lambda w: -w[1])

    # return all the possibilities sorted by probability
    return result

def get_hypernyms(word):
    hyper = []

    for i,j in enumerate(wn.synsets(word)):
        for l in j.hypernyms():
            hyper += l.lemma_names()
            
    return set(hyper)

def get_cats_dict():
    cats = ['running', 'dancing', 'bicycling', 'walking', 'hunting', 'ball',
        'standing', 'sitting', 'skiing', 'swimming', 'cooking', 'driving', 'climbing', 
        'horseback', 'skateboarding', 'yoga', 'canoe', 'training', 'lying']
    
    cats_dict = dict(zip(cats, [{"keywords": {cats[i]}, 
                             "model": './models/out_' + cats[i] + '/final.pt', 
                             "dataset": MPIIDataset('../data/prepared_data/mpii_' + cats[i] + '.json')} 
                            for i in range(len(cats))]))
    hypernyms_cats = ['bicycling', 'horseback', 'skateboarding', 'canoe']

    for cat in hypernyms_cats:
        cats_dict[cat]["keywords"] = cats_dict[cat]["keywords"].union(get_hypernyms(cat))
        
    for cat in cats:
        morps = morphify(cat,'v','n')
        good_morps = []
        for i in range(len(morps)):
            if (morps[i][1] > 0.1):
                good_morps.append(morps[i][0])
        cats_dict[cat]["keywords"] = cats_dict[cat]["keywords"].union(set(good_morps))
        
    return cats_dict

def get_generated_pose(model):
    checkpoint = torch.load(model)
    gen = Generator().to('cuda')
    gen.load_state_dict(checkpoint['g_state_dict'])
    label = dataset.one_hot[0]
    label = Variable(label.type(torch.cuda.LongTensor))
    noise = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (2, 32))), requires_grad=False)
    out = gen(noise, label).cpu().detach().numpy()
    
    return out[0].tolist()

def get_dataset_pose(dataset):
    len_data = len(dataset)
    return dataset[randrange(len_data - 1)][0].numpy().tolist()

def get_poses_by_text(text, cats_dict):
    res = []
    
    for cat in cats_dict.keys():
        words_text = preprocess(text)
        for word in cats_dict[cat]["keywords"]:
            if (word in words_text):
                flag = randrange(0, 2)
                if (cat == 'standing' or 'lying') and flag:
                    res = get_generated_pose([cats_dict[cat]["model"]])
                else: 
                    res = get_dataset_pose(cats_dict[cat]["dataset"])

                return res
    
    flag = randrange(0, 2)
    res = get_generated_pose([cats_dict[cat]["model"]]) if flag else get_dataset_pose(cats_dict[cat]["dataset"])
    
    return res

descr = "This is a program that generates pose by a text description."

parser = argparse.ArgumentParser(description=descr)
parser.add_argument("--text", "-t", help="set text description")

args = parser.parse_args()

cats_dict = get_cats_dict()

if args.text:
    text = args.text
    poses = get_poses_by_text(text, cats_dict)
    print(poses)
else:
    raise ValueError("Missing text description.")
