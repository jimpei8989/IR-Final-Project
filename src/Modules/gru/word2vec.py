import os
import numpy as np
import pandas as pd
import argparse
import gensim
from tensorflow.keras.preprocessing.text import text_to_word_sequence

import utils

class Word2Vec():
    def __init__(self, embedding_dim=None):
        self.fitted = False
        self.embedding_dim = embedding_dim
        self.model = None

    def load(self, model_path):
        self.fitted = True
        self.model = gensim.models.Word2Vec.load(model_path)
        self.embedding_dim = self.model.vector_size
        return self

    def fit(self, X):
        self.model = gensim.models.Word2Vec(X, size=self.embedding_dim, window=5, min_count=5, workers=os.cpu_count(), iter=10, sg=1)
        self.fitted = True
        return self

    def save(self, model_path):
        if not self.fitted:
            raise NotImplementedError
        self.model.save(model_path)

    def get_embedding(self):
        embedding = [self.model[w] for w in self.model.wv.vocab]
        pad_vector = [0.] * self.embedding_dim
        pad_vector[0] = 1e-5
        unk_vector = [0.] * self.embedding_dim
        unk_vector[1] = 1e-5
        embedding.extend([pad_vector, unk_vector])
        return np.array(embedding, np.float32)

    def get_word2idx(self):
        word2idx = {w:i for i, w in enumerate(self.model.wv.vocab)}
        word2idx['<PAD>'] = len(word2idx)
        word2idx['<UNK>'] = len(word2idx)
        return word2idx

