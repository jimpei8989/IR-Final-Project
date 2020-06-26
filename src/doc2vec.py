import os, argparse

import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from gensim.parsing.porter import PorterStemmer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity

from Modules import utils
from dataset import QueryDataset

def get_similar(model, query, topk=100, batch_size=2500):
    query_vec = np.array([model.infer_vector(q.split()) for q in query])
    print(query_vec.shape)
    if isinstance(query, str):
        sims = cosine_similarity(query_vec.reshape(1, -1), model.docvecs.vectors_docs)[0]
        arg = np.argpartition(sims, -topk)[-topk:]
        return arg[np.argsort(sims[arg])[::-1]]

    print(f'query_vec: {query_vec.shape}')
    topk_idx = []
    for i in trange(0, len(query_vec), batch_size):
        sims = cosine_similarity(query_vec[i:i + batch_size], model.docvecs.vectors_docs)
        arg = np.argpartition(sims, -topk, axis=1)[:, -topk:]
        topk_idx.append(np.take_along_axis(arg, np.argsort(np.take_along_axis(sims, arg, axis=1), axis=1)[:, ::-1], axis=1))

    return np.concatenate(topk_idx, axis=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('-dt', '--docs-train-file')
    parser.add_argument('-di', '--docid-pkl-file')
    parser.add_argument('-q', '--query-dirs', nargs=3, help='training query dir, dev query dir, test query, dir')
    parser.add_argument('-vs', '--vector-size', type=int, default=100)
    parser.add_argument('-s', '--stemming', action='store_true')
    args = parser.parse_args()
    
    assert not (bool(args.query_dirs) ^ bool(args.docid_pkl_file)), 'docid_pkl_file must exist if query_dirs exist.'
        
    if os.path.exists(args.model_path):
        print('\033[32;1mLoading Model...\033[0m')
        doc2vec = Doc2Vec.load(args.model_path)
    else:
        doc2vec = Doc2Vec(corpus_file=args.docs_train_file, vector_size=args.vector_size, workers=os.cpu_count())
        doc2vec.save(args.model_path)

    if args.query_dirs:
        idx2id = utils.pickleLoad(args.docid_pkl_file)
        for query_dir, task in zip(args.query_dirs, ['Training', 'Development', 'Testing']):
            query_set = QueryDataset(query_dir)

            if args.stemming:
                queries = PorterStemmer().stem_documents(query_set.queries[:100])
            else:
                queries = query_set.queries[:10000]
            pred_idx = get_similar(doc2vec, queries, 1000)
            pred_id = idx2id[pred_idx]
            print(f'\033[32;1m{task} MAP Score: {utils.MAP(query_set.relevantDocuments[:10000], pred_id)}\033[0m')