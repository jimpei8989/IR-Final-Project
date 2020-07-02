import os

class Corpus:
    def __init__(self, corpusDir, docIDFile, usedFeatures=['title', 'content']):
        self.usedFeatures = usedFeatures
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
        features = []
        if 'title' in self.usedFeatures: features.append(2)
        if 'content' in self.usedFeatures: features.append(3)

        def getDocument(docID):
            corpusFD.seek(self.offsetLookup[docID])
            sections = corpusFD.readline().strip().split('\t')
            if len(sections) == 4:
                return ' '.join(s for i, s in enumerate(sections) if i in features)
            else:
                return ''
                

        for docID in self.documentIDs:
            yield getDocument(docID)

        corpusFD.close()

    def __len__(self):
        return len(self.documentIDs)

    def idx2DocID(self, idx):
        return self.documentIDs[idx]

class QueryDataset:
    def __init__(self, queryDir, num=None):
        with open(os.path.join(queryDir, 'queries.tsv')) as f:
            self.queryIDs, self.queries = zip(*map(lambda line : line.strip().split('\t'), f.readlines()))
        
            self.ID2idx = {q : i for i, q in enumerate(self.queryIDs)}

        with open(os.path.join(queryDir, 'topK.csv')) as f:
            self.relevantDocuments = [line.strip().split(',')[1].split(' ') for line in f.readlines()]

        if num is not None:
            self.queryIDs = self.queryIDs[:num]
            self.queries = self.queries[:num]
            self.relevantDocuments = self.relevantDocuments[:num]    

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
