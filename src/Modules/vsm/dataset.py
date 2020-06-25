import os

class Corpus:
    def __init__(self, corpusDir, docIDFile):
        with open(docIDFile) as f:
            self.documentIDs = [line.strip() for line in  f.readlines()]
            self.docID2idx = {docID : idx for idx, docID in enumerate(self.documentIDs)}

        self.offsetLookup = {}
        with open(os.path.join(corpusDir, 'msmarco-docs-lookup.tsv')) as f:
            for docID, trecOffset, tsvOffset in map(lambda line : line.strip().split(), f.readlines()):
                self.offsetLookup[docID] = int(tsvOffset)

        # Get the corresponding document
        self.corpusFile = os.path.join(corpusDir, 'msmarco-docs.tsv')

    def __iter__(self):
        corpusFD = open(self.corpusFile)
        def getDocument(docID):
            corpusFD.seek(self.offsetLookup[docID])
            try:    # Some documents might not be saved well
                docID, url, title, content = corpusFD.readline().strip().split('\t')
                return title + ' ' + content
            except:
                return ''

        for docID in self.documentIDs:
            yield getDocument(docID)

        corpusFD.close()

    def __len__(self):
        return len(self.documentIDs)

    def idx2DocID(self, idx):
        return self.documentIDs[idx]

class QueryDataset:
    def __init__(self, queryDir):
        with open(os.path.join(queryDir, 'queries.tsv')) as f:
            self.queryIDs, self.queries = zip(*map(lambda line : line.strip().split('\t'), f.readlines()))
            self.ID2idx = {q : i for i, q in enumerate(self.queryIDs)}

        with open(os.path.join(queryDir, 'topK.csv')) as f:
            self.relevantDocuments = [line.strip().split(',')[1].split(' ') for line in f.readlines()]

    def __iter__(self):
        for query, rel in zip(self.queries, self.relevantDocuments):
            yield query, rel

    def __len__(self):
        return len(self.queryIDs)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.queries[idx], self.relevantDocuments[idx]
        elif isinstance(idx, str):
            idx = self.ID2idx[idx]
            return self.queries[idx], self.relevantDocuments[idx]
