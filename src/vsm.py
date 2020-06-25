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

def main():
    args = parseeArguments()

    # Get the corpus generator
    corpus = Corpus(os.path.join(args.dataDir, 'corpus/'), args.docIDFile)

    if args.fit:
        with EventTimer('Fitting model'):
            # Create Model
            model = TfidfVectorizer(
                stop_words='english',
                max_features =args.maxFeatures,
                max_df=args.maxDF,
                min_df=args.minDF,
                sublinear_tf=args.sublinearTF
            )

            tfidf = model.fit_transform(tqdm(corpus))
            utils.pickleSave(model, os.path.join(args.modelDir, 'model.pkl'))
            scipy.sparse.save_npz(os.path.join(args.modelDir, 'tfidf.npz'), tfidf)
    else:
        with EventTimer('Load model'):
            model = utils.pickleLoad(os.path.join(args.modelDir, 'model.pkl'))
            tfidf = scipy.sparse.load_npz(os.path.join(args.modelDir, 'tfidf.npz'))

    print(f'> TFIDF shape: {tfidf.shape}')

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
                print(f'{qdir} -> MAP: {np.mean(APs):.4f}', file=f)

def parseeArguments():
    parser = ArgumentParser()
    parser.add_argument('--dataDir', default='data/')
    parser.add_argument('--docIDFile', default='data/partial/corpus/docIDs')
    parser.add_argument('--queryDir', nargs='+', default=['data/partial/train/'])
    parser.add_argument('--modelDir', default='models/vsm/')
    parser.add_argument('--fit', action='store_true')
    parser.add_argument('--maxFeatures', type=int, default=10**5, help='Maximum features used')
    parser.add_argument('--minDF', type=int, default=100, help='Minimum appearance can a word be adopted')
    parser.add_argument('--maxDF', type=float, default=0.5, help='Maximum frequency can a word be adopted')
    parser.add_argument('--sublinearTF', type=bool, default=True)
    parser.add_argument('--topK', type=int, default=1000)
    parser.add_argument('--numWorkers', type=int, default=8)
    return parser.parse_args()

if __name__ == '__main__':
    main()
