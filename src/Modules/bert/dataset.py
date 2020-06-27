import os
from collections import deque

import numpy as np
import torch
from torch.utils.data import Dataset


class Random:
    def __init__(self, where=(0, 2), size = 1):
        if isinstance(where, tuple):
            self.where = np.arange(where[0], where[1])
        else:
            self.where = where

        self.size = size
        self.buffer = deque()

    def __call__(self):
        if len(self.buffer) == 0:
            self.buffer.extend(np.random.choice(self.where, self.size))
        return self.buffer.popleft()

class RetrievalDataset(Dataset):
    def __init__(self, corpusDir, docIDFile, queryDir):
        super().__init__()

        # Build Corpus
        with open(docIDFile) as f:
            self.documentIDs = [line.strip() for line in  f.readlines()]
            self.docID2idx = {docID : idx for idx, docID in enumerate(self.documentIDs)}

        self.offsetLookup = {}
        with open(os.path.join(corpusDir, 'msmarco-docs-lookup.tsv')) as f:
            for docID, trecOffset, tsvOffset in map(lambda line : line.strip().split(), f.readlines()):
                self.offsetLookup[docID] = int(tsvOffset)

        # Get the corresponding document path
        self.corpusFD = open(os.path.join(corpusDir, 'msmarco-docs.tsv'))

        # Build query
        with open(os.path.join(queryDir, 'queries.tsv')) as f:
            self.queryIDs, self.queries = zip(*map(lambda line : line.strip().split('\t'), f.readlines()))
            self.queryID2idx = {q : i for i, q in enumerate(self.queryIDs)}

        with open(os.path.join(queryDir, 'topK.csv')) as f:
            self.relDocIDs = [set(line.strip().split(',')[1].split(' ')) for line in f.readlines()]
            self.positiveSampler = [Random(list(posSet)) for posSet in self.relDocIDs]

        self.epochStepSize = len(self.queries) * 5

        self.queryRdGenerator = Random((0, len(self.queries)), self.epochStepSize)
        self.docRdGenerator = Random((0, len(self.documentIDs)), self.epochStepSize)
        self.posNegSampler = Random((0, 5), self.epochStepSize)

    def __del__(self):
        self.corpusFD.close()

    def __getitem__(self, idx):
        def getDocument(docID):
            self.corpusFD.seek(self.offsetLookup[docID])
            sections = self.corpusFD.readline().strip().split('\t')
            if len(sections) == 4:
                return ' '.join(sections[2:]).lower()
            else:
                return ''
                
        queryIdx = self.queryRdGenerator()
        positive = self.posNegSampler() == 0

        if positive:
            docIdx = self.positiveSampler[queryIdx]()
        else:
            while True:
                docIdx = self.docRdGenerator()
                if docIdx not in self.relDocIDs[queryIdx]:
                    docIdx = self.documentIDs[docIdx]
                    break

        return getDocument(docIdx), self.queries[queryIdx], int(positive)

    def __len__(self):
        return self.epochStepSize
