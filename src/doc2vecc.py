import os, argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim.parsing.porter import PorterStemmer
from joblib import Parallel, delayed

from Modules import utils
from Modules.Doc2VecC_python.doc2vecc import Doc2VecC
from dataset import QueryDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('-dp', '--docs-pkl-file')
    parser.add_argument('-o', '--output-train-file')
    parser.add_argument('-wv', '--wordvec-file')
    parser.add_argument('-dv', '--docvec-file')
    parser.add_argument('-v', '--vocab-file')
    parser.add_argument('-di', '--docid-pkl-file')
    parser.add_argument('-q', '--query-dirs', nargs=3, help='training query dir, dev query dir, test query, dir')
    parser.add_argument('-vs', '--vector-size', type=int, default=128)
    parser.add_argument('-s', '--stemming', action='store_true')
    args = parser.parse_args()
    
    assert not (bool(args.query_dirs) ^ bool(args.docid_pkl_file)), 'docid_pkl_file must exist if query_dirs exist.'
        
    if os.path.exists(args.model_path):
        print('\033[32;1mLoading Model...\033[0m')
        doc2vecc = Doc2VecC.load(args.model_path)
    else:
        doc2vecc = Doc2VecC(args.output_train_file, args.wordvec_file, args.docvec_file, size=args.vector_size, threads=os.cpu_count(), vocab_file=args.vocab_file)
        docs = utils.pickleLoad(args.docs_pkl_file)

        if args.stemming:
            with utils.EventTimer('Begin stemming'):
                docs = PorterStemmer().stem_documents(docs)
        doc2vecc.fit(docs)
        doc2vecc.save(args.model_path)

    if args.query_dirs:
        idx2id = utils.pickleLoad(args.docid_pkl_file)
        for query_dir, task in zip(args.query_dirs, ['Training', 'Development', 'Testing']):
            query_set = QueryDataset(query_dir)

            if args.stemming:
                queries = PorterStemmer().stem_documents(query_set.queries[:100])
            else:
                queries = query_set.queries
            pred_idx = doc2vecc.get_similar(queries, 1000)
            pred_id = idx2id[pred_idx]
            print(f'\033[32;1m{task} MAP Score: {utils.MAP(query_set.relevantDocuments[:100], pred_id)}\033[0m')