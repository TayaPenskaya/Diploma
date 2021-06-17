import string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from itertools import chain
from collections import defaultdict
from nltk.corpus import wordnet as wn

import nltk
nltk.download("wordnet")
nltk.download("stopwords")

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
from os.path import isfile

import numpy as np

class Text2Pose:
    
    def __init__(self):
        self.cats = ['running', 'dancing', 'bicycling', 'walking', 'hunting', 'ball',
            'standing', 'sitting', 'skiing', 'swimming', 'cooking', 'driving', 'climbing', 
            'horseback', 'skateboarding', 'yoga', 'canoe', 'training', 'lying']
        self.hypernyms_cats = ['bicycling', 'horseback', 'skateboarding', 'canoe']
        self.cats_dict = self.get_cats_dict()
        self.text = ''
        
    def get_pose_by_text(self, text):
        self.text = text
        res = []
        words_text = Text2Pose.preprocess(self.text)
        
        for cat in self.cats_dict.keys():
            for word in self.cats_dict[cat]["keywords"]:
                if (word in words_text):
                    flag = randrange(0, 2)
                    if (cat == 'standing' or 'lying') and flag:
                        res = Text2Pose.__get_generated_pose(self.cats_dict[cat]["model"])
                    else: 
                        res = Text2Pose.__get_dataset_pose(self.cats_dict[cat]["dataset"])

                    return res

        flag = randrange(0, 2)
        res = Text2Pose.__get_generated_pose(self.cats_dict["standing"]["model"]) if flag \
              else Text2Pose.__get_dataset_pose(self.cats_dict["standing"]["dataset"])

        return res
    
    def get_cats_dict(self):    
        cats_dict = dict(zip(self.cats, [{"keywords": {self.cats[i]}, 
                                 "model": './models/out_' + self.cats[i] + '/final.pt', 
                                 "dataset": MPIIDataset('./data/prepared_data/mpii_' + self.cats[i] + '.json')} 
                                for i in range(len(self.cats))]))

        for cat in self.hypernyms_cats:
            cats_dict[cat]["keywords"] = cats_dict[cat]["keywords"].union(Text2Pose.get_hypernyms(cat))
            
        for cat in self.cats:
            morps = Text2Pose.morphify(cat,'v','n')
            good_morps = []
            for i in range(len(morps)):
                if (morps[i][1] > 0.1):
                    good_morps.append(morps[i][0])
            cats_dict[cat]["keywords"] = cats_dict[cat]["keywords"].union(set(good_morps))
        
        for cat in self.cats:
            if isfile(cats_dict[cat]["model"]):
                cats_dict[cat]["model"] = torch.load(cats_dict[cat]["model"], map_location=torch.device('cpu'))

        return cats_dict
    
    @staticmethod
    def __get_dataset_pose(dataset):
            len_data = len(dataset)
            return dataset[randrange(len_data - 1)][0].numpy().tolist() 
    
    @staticmethod
    def __get_generated_pose(model): 
            gen = Generator().to('cpu')
            gen.load_state_dict(model['g_state_dict'])
            noise = Variable(torch.FloatTensor(np.random.normal(0, 1, (2, 32))), requires_grad=False)
            out = gen(noise,  Variable(torch.LongTensor([1], device='cpu'))).cpu().detach().numpy()
            return out[0].tolist()

    @staticmethod
    def morphify(word, org_pos, target_pos):
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

    @staticmethod
    def get_hypernyms(word):
        hyper = []

        for i,j in enumerate(wn.synsets(word)):
            for l in j.hypernyms():
                hyper += l.lemma_names()

        return set(hyper)
    
    @staticmethod
    def preprocess(text):
        stoplist = list(set(stopwords.words('english')))
        punctuation = list(string.punctuation)
        tokenizer = RegexpTokenizer(r'\w+')
        lemmatizer = WordNetLemmatizer()

        final_text = []
        for i in tokenizer.tokenize(text):
            word = i.strip().lower() 
            if not (word in stoplist or word in punctuation or len(word) < 3):
                final_text.append(lemmatizer.lemmatize(word))
        return list(set(final_text))
    
if __name__ == '__main__':
    
    descr = "This is a program that generates pose by a text description."

    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument("--text", "-t", help="set text description")

    args = parser.parse_args()

    if args.text:
        t2p = Text2Pose()
        poses = t2p.get_pose_by_text(args.text)
        print(poses)
    else:
        raise ValueError("Missing text description.")
