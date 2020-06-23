import os, sys
import typing
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from joblib import Parallel, delayed

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from Modules.utils import EventTimer

class Model():
    def __init__(self, maxFeatures=None, maxDF=1.0, minDF=1, numWorkers=8):
        self.maxFeatures = maxFeatures
        self.maxDF = maxDF
        self.minDF = minDF
        self.numWorkers = numWorkers

    def build(self, corpusDir, docIDFile):
        with open(docIDFile) as f:
            self.documentIDs = map(lambda line: line.strip(), f.readlines())

        offsetLookup = {}
        with open(os.path.join(corpusDir, 'msmarco-docs-lookup.tsv')) as f:
            for QID, trecOffset, tsvOffset in map(lambda line : line.strip().split(), f.readlines()):
                offsetLookup[QID] = int(tsvOffset)

        with open(os.path.join(corpusDir, 'msmarco-docs.tsv')) as f:
            def getDocument(docID):
                f.seek(offsetLookup[docID])
                docID, url, title, content = f.readline().split('\t')
                return title + content

            corpus = list(map(getDocument, tqdm(self.documentIDs)))

        with EventTimer('Stemming'):
            stemmer = PorterStemmer()
            # corpus = Parallel(n_jobs=8, backend='threading', verbose=5)(delayed(stemmer.stem)(doc) for doc in corpus)
            with ThreadPool(self.numWorkers) as p:
                corpus = p.map(stemmer.stem, tqdm(corpus))

        # Use nltk stemming and tokenizer
        # Ignore words appear in > 0.5 documents, and <0.05 documents
        self.model = TfidfVectorizer(
            stop_words='english',
            max_features=self.maxFeatures,
            max_df = self.maxDF,
            min_df = self.minDF
        )

        with EventTimer('Fitting tfidf'):
            tfidf = self.model.fit_transform(tqdm(corpus))
            print(f'> TF-IDF shape: {tfidf.shape}')

        return tfidf

    def transform(self, queries: typing.List[str]):
        with EventTimer('Stemming (transform)'):
            stemmer = PorterStemmer()

            with ThreadPool(self.numWorkers) as p:
                queries = p.map(stemmer.stem, tqdm(queries))

        with EventTimer('Transform'):
            queries = self.model.transform()
        
        return queries

    def save(self, modelDir):
        os.makedirs(modelDir, exist_ok=True)
        utils.pickleSave(self.model, os.path.join(modelDir, 'model.pkl'))

    def load(self, modelDir):
        self.model = utils.pickleLoad(os.path.join(modelDir, 'model.pkl'))
