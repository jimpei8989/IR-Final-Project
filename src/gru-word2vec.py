import os
import numpy as np
import pandas as pd
import argparse
import gensim
from tensorflow.keras.preprocessing.text import text_to_word_sequence

from Modules.gru.word2vec import Word2Vec
from Modules import utils
from dataset import Corpus, QueryDataset

def split_doc2sent(docs):
    for doc in docs:
        for sent in gensim.summarization.textcleaner.get_sentences(doc):
            yield text_to_word_sequence(sent)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('train_pkl')
    parser.add_argument('test_data')
    args = parser.parse_args()

    trainX = utils.pickleLoad(args.train_pkl)

    print("loading testing data ...")
    testX = utils.load_test_data(args.test_data, preprocessing=False)

    model = Word2Vec(256).fit(trainX + testX)
    
    print("saving model ...")
    model.save(args.model_path)
