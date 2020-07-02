import os, argparse

import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from gensim.parsing.porter import PorterStemmer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.preprocessing import normalize


from Modules import utils
from dataset import QueryDataset

def get_similar(model, query, topk=100, batch_size=2500, relevance_feedback_steps=2, alpha=0.8, relevance_doc_num=100):
    query_vec = np.array([model.infer_vector(q.split()) for q in query])
    print(query_vec.shape)

    docvecs_norm = normalize(model.docvecs.vectors_docs)
    if isinstance(query, str):
        for _ in range(relevance_feedback_steps):               
            sims = (docvecs_norm @ query_vec.reshape(-1, 1)).ravel()
            arg = np.argpartition(sims, -topk)[-topk:]
            arg_sorted = arg[np.argsort(sims[arg])[::-1]]
            query_vec = alpha * query_vec + (1 - alpha) * np.mean(docvecs_norm[arg_sorted[:relevance_doc_num]], axis=0)
        return arg_sorted

    print(f'query_vec: {query_vec.shape}')
    topk_idx = []
    for i in trange(0, len(query_vec), batch_size):
        for _ in range(relevance_feedback_steps):               
            sims = query_vec[i:i + batch_size] @ docvecs_norm.T
            arg = np.argpartition(sims, -topk, axis=1)[:, -topk:]
            arg_sorted = np.take_along_axis(arg, np.argsort(np.take_along_axis(sims, arg, axis=1), axis=1)[:,::-1], axis=1)
            rel_vec = np.mean(docvecs_norm[arg_sorted[:,:relevance_doc_num]], axis=1)
            query_vec[i:i + batch_size] = alpha * query_vec[i:i + batch_size] + (1 - alpha) * rel_vec
        topk_idx.append(arg_sorted)

    return np.concatenate(topk_idx, axis=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('-dt', '--docs-train-file')
    parser.add_argument('-di', '--docid-pkl-file')
    parser.add_argument('-q', '--query-dirs', nargs=3, help='training query dir, dev query dir, test query, dir')
    parser.add_argument('-vs', '--vector-size', type=int, default=100)
    parser.add_argument('-s', '--stemming', action='store_true')
    parser.add_argument('-r', '--relevance-feedback-steps', type=int, default=2)
    parser.add_argument('-a', '--alpha', type=float, default=0.8, help='Original ratio for relevance feedback.')
    parser.add_argument('-rn', '--relevance-doc-num', type=int, default=100, help='Number of documents to be treat as relevant ones during relevance feedback.')

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
                queries = query_set.queries
            pred_idx = get_similar(doc2vec, queries, 1000, relevance_feedback_steps=args.relevance_feedback_steps, alpha=args.alpha, relevance_doc_num=args.relevance_doc_num)
            pred_id = idx2id[pred_idx]
            print(f'\033[32;1m{task} MAP Score: {utils.MAP(query_set.relevantDocuments, pred_id)}\033[0m')