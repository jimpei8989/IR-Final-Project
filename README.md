# Information Retrieval - Final Project

> Web Retrieval and Mining 2020 (CSIE 5137)

**Topic**: <u>*Information Retrieval using SOTA Neural Network Models*</u>

**Team Name**: <u>賈裴根陳尹症候群</u>

**Team Members**: 陳義榮 (B06902001), 裴梧鈞 (B06902029), 賈本晧 (B06902039), 尹聖翔 (B06902103)

### Useful Links

- [Hackmd - Memo](https://hackmd.io/@jimpei8989/SyzKDbva8)

### Data Splitting
- Corpus: Randomly Select 1M documents
    - `data/partial/corpus/docIDs`, each line is a document we chose. *shasum: a6b8f150e0fa424614ecc1f636dbbe74cd62a1db*
- Train
    - Queries: 50K queries
    - `data/partial/train/queries.tsv`, each line is "queryID<TAB>query string". *shasum: 6365ca4dee17a0c7d7719af42be367c1294ac763*
    - `data/partial/train/topK.csv`, each line is "queryID,relevantDocuments"; the relevant documents are seperated with spaces
- Development
    - Similar to `data/partial/train/topK.csv`
- Test
    - Similar to `data/partial/train/topK.csv`
