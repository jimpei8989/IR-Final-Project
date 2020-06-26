import os
from argparse import ArgumentParser
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from Modules.utils import EventTimer
from Modules import utils
from Modules.vsm.dataset import Corpus, QueryDataset

import nltk
from nltk import PorterStemmer, word_tokenize

def StemmingTokenizer(text):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in word_tokenize(text)]

def main():
    args = parseeArguments()

    # Get the corpus generator
    corpus = Corpus(os.path.join(args.dataDir, 'corpus/'), args.docIDFile, args.usedFeatures)

    with EventTimer('Calculate Document Length'):
        documentLengthes = np.array(list(map(lambda doc : len(doc.split(' ')), corpus)))
        avgDL = np.mean(documentLengthes)
        print(f'> avgDL:\t{avgDL}')

    with EventTimer('Load model'):
        model = utils.pickleLoad(os.path.join(args.modelDir, 'model.pkl'))
        tfidf = scipy.sparse.load_npz(os.path.join(args.modelDir, 'tfidf.npz'))

        idf = model.idf_.reshape(1, -1)
        tf = tfidf.multiply(scipy.sparse.csr_matrix(1 / idf))

        # TODO: convert to the tfidf
        documentLengthes = scipy.sparse.csr_matrix(documentLengthes.reshape(-1, 1))
        tf = (tf * (args.k1 + 1)) / (tf + args.k1 * (1 - args.b + args.b * documentLengthes / avgDL))

    print(f'> TFIDF shape: {tfidf.shape}')
    print(f'> IDF shape: {idf.shape}')

    # Query
    def evaluate(q):
        query, rel = q
        queryVec = model.transform([query]).toarray()
        scores = (tfidf @ queryVec.reshape(-1, 1)).reshape(-1)
        topKIdxes = np.argpartition(-scores, args.topK)[:args.topK]
        topKIdxes = sorted(topKIdxes, key=lambda i:scores[i], reverse=True)
        return utils.MAP([rel], [map(corpus.idx2DocID, topKIdxes)])

    with open(os.path.join(args.modelDir, 'result.txt'), 'w') as f:
        for qdir in args.queryDir:
            with EventTimer(f'Query on {qdir}'):
                queries = QueryDataset(qdir)

                with ThreadPool(args.numWorkers) as p:
                    APs = list(tqdm(p.imap(evaluate, queries), total=len(queries)))

                print(f'> MAP: {np.mean(APs):.4f}')
                print(f'{qdir}\t-> MAP: {np.mean(APs):.4f}', file=f)

def parseeArguments():
    parser = ArgumentParser()
    parser.add_argument('--dataDir', default='data/')
    parser.add_argument('--docIDFile', default='data/partial/corpus/docIDs')
    parser.add_argument('--queryDir', nargs='+', default=['data/partial/train/'])
    parser.add_argument('--modelDir', default='models/vsm/')
    parser.add_argument('--usedFeatures', nargs='+', default=['title', 'content'])
    parser.add_argument('--k1', type=float, default=1.2)
    parser.add_argument('--b', type=float, default=0.75)
    parser.add_argument('--topK', type=int, default=1000)
    parser.add_argument('--numWorkers', type=int, default=8)
    return parser.parse_args()

if __name__ == '__main__':
    main()
