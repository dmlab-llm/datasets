import argparse
import math
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import time
import gzip
import glob
import torch
import numpy as np
import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from transformers import GPT2Tokenizer
from transformers import AutoTokenizer

import math
import numpy as np
import itertools
from scipy.special import softmax
import matplotlib.pyplot as plt
import pandas as pd

def entropy(tokens):
    count_token = [len(list(y)) for x, y in itertools.groupby(tokens)]
    count_token = np.array(count_token)
    length = len(tokens)

    e = 0.0
    for val in count_token:
        p = val / length
        e += p * math.log2(1/p)

    return e

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
data_path = "/root/datasets/squad/train.csv"

stop_words = set(stopwords.words("english"))

#data_list = []

data_list = pd.read_csv(data_path)

tokenized_data = []
ids = []
entropies = []
cnt = 0

for context, question in zip(data_list['context'], data_list['question']):
    text = context + "\n" + question
    raw_tokens = tokenizer.tokenize(text, truncation=True, max_length=3000)
    tokens = []
    for token in raw_tokens:
        if token.lower() not in stop_words and token not in ",.'!?" and token != '"':
            tokens.append(token)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    tokenized_data.append(tokens)

    ids.append(ids)
    entropies.append(entropy(ids))
    if cnt % 100000 == 0:
        print(cnt, tokens, ids)
    cnt += 1

#print(entropies)

data_list["entropy"] = entropies

data_list.sort_values(by="entropy", axis = 0, inplace = True)
data_list.to_csv("/root/datasets/squad/train_max_en_up.csv")

data_list.sort_values(by="entropy", axis=0, ascending=False, inplace=True)
data_list.to_csv("/root/datasets/squad/train_max_en_down.csv")
