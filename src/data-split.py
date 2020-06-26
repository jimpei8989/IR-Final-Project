import os, shutil
import numpy as np
from tqdm import tqdm
import pickle

from argparse import ArgumentParser
from Modules.utils import EventTimer
from Modules import utils

def main():
    np.random.seed(utils.SEED)

    args = parseArguments()

    # mkdir is does not exists
    os.makedirs(args.outputDir, exist_ok=True)
    os.makedirs(os.path.join(args.outputDir, 'corpus'), exist_ok=True)
    os.makedirs(os.path.join(args.outputDir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.outputDir, 'dev'), exist_ok=True)
    os.makedirs(os.path.join(args.outputDir, 'test'), exist_ok=True)

    # Get all corpus documents
    with EventTimer('Get all corpus documents'):
        with open('data/corpus/msmarco-docs-lookup.tsv') as f:
            # Format: DocID, TREC offset, TSV Offset
            corpus = list(map(lambda line : line.split()[0], f.readlines()))
            print(f'> # all document ids: {len(corpus)}')
    
    # Randomly choice 1M documents
    with EventTimer('Choosing the 1M docments'):
        partialCorpusIDs = np.random.choice(corpus, size=args.corpusSize, replace=False)
        partialCorpusSet = set(partialCorpusIDs)

        assert len(partialCorpusIDs) == len(partialCorpusSet)

        with open(os.path.join(args.outputDir, 'corpus', 'docIDs'), 'w') as f:
            print('\n'.join(partialCorpusIDs), file = f)

    with EventTimer('Handling Training Data'):
        # Get all queries
        with EventTimer('Choosing the 50K queries'):
            with open('data/train/msmarco-doctrain-queries.tsv') as f:
                data = map(lambda line : line.strip().split('\t'), f.readlines())
                queryTitles = {i : t for i, t in data}
                queryIDs = list(queryTitles.keys())

            # Randomly select 50k
            partialQueryIDs = np.random.choice(queryIDs, size=args.querySize, replace=False)
            partialQuerySet = set(partialQueryIDs)

        relevantDocuments = dict()
        with open('data/train/msmarco-doctrain-top100') as f:
            for line in tqdm(f.readlines()):
                queryID, _, docID, _, _, _ = line.split()
                if queryID in partialQuerySet:
                    if queryID not in relevantDocuments:
                        relevantDocuments[queryID] = list()
                    if docID in partialCorpusSet:
                        relevantDocuments[queryID].append(docID)

        with open(os.path.join(args.outputDir, 'train', 'queries.tsv'), 'w') as f:
            for QID in partialQueryIDs:
                print(f'{QID}\t{queryTitles[QID]}', file=f)

        with open(os.path.join(args.outputDir, 'train', 'topK.csv'), 'w') as f:
            for QID in partialQueryIDs:
                print(f'{QID},{" ".join(relevantDocuments[QID])}', file=f)

    with EventTimer('Handling Development Data'):
        relevantDocuments = dict()
        with open('data/dev/msmarco-docdev-top100') as f:
            for line in tqdm(f.readlines()):
                queryID, _, docID, _, _, _ = line.split()
                if queryID not in relevantDocuments:
                    relevantDocuments[queryID] = list()
                if docID in partialCorpusSet:
                    relevantDocuments[queryID].append(docID)

        # Copy queries
        shutil.copyfile('data/dev/msmarco-docdev-queries.tsv', os.path.join(args.outputDir, 'dev', 'queries.tsv'))

        with open(os.path.join(args.outputDir, 'dev', 'topK.csv'), 'w') as f:
            for queryID, docIDs in relevantDocuments.items():
                print(f'{queryID},{" ".join(docIDs)}', file = f)
                
    with EventTimer('Handling Testing Data'):
        relevantDocuments = dict()
        with open('data/test/msmarco-doctest2019-top100') as f:
            for line in tqdm(f.readlines()):
                queryID, _, docID, _, _, _ = line.split()
                if queryID not in relevantDocuments:
                    relevantDocuments[queryID] = list()
                if docID in partialCorpusSet:
                    relevantDocuments[queryID].append(docID)

        # Copy queries
        shutil.copyfile('data/test/msmarco-test2019-queries.tsv', os.path.join(args.outputDir, 'test', 'queries.tsv'))

        with open(os.path.join(args.outputDir, 'test', 'topK.csv'), 'w') as f:
            for queryID, docIDs in relevantDocuments.items():
                print(f'{queryID},{" ".join(docIDs)}', file=f)
                
    if args.generateCorpus:
        with EventTimer('Generating Corpus'):
            offsetLookup = {}
            with open(os.path.join('data', 'corpus', 'msmarco-docs-lookup.tsv')) as f:
                for QID, trecOffset, tsvOffset in map(lambda line : line.strip().split(), f.readlines()):
                    offsetLookup[QID] = int(tsvOffset)

            with open(os.path.join('data', 'corpus', 'msmarco-docs.tsv')) as f:
                def getDocument(docID):
                    f.seek(offsetLookup[docID])
                    docID, url, title, content = f.readline().split('\t')
                    return title + content
                corpus = list(map(getDocument, tqdm(partialCorpusIDs)))
        
        with open('data/partial/corpus/partial_corpus.pkl', 'wb') as f:
            pickle.dump(corpus, f)
        with open('data/partial/corpus/docIDs.pkl', 'wb') as f:
            pickle.dump(partialCorpusIDs, f)

        
def parseArguments():
    parser = ArgumentParser()
    parser.add_argument('--outputDir', default='data/partial')
    parser.add_argument('--corpusSize', type=int, default=1000000)
    parser.add_argument('--querySize', type=int, default=50000)
    parser.add_argument('--generateCorpus', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    main()
