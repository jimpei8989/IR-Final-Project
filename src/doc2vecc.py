import os, argparse
import numpy as np
from time import time
import pandas as pd
from Modules import utils
from sklearn.metrics.pairwise import cosine_similarity
import pickle

from Modules.Doc2VecC_python.doc2vecc import Doc2VecC

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    parser.add_argument('docs_pkl_file')
    parser.add_argument('output_train_file')
    parser.add_argument('wordvec_file')
    parser.add_argument('docvec_file')
    parser.add_argument('vocab_file')
    parser.add_argument('-v', '--vector-size', type=int, default=128)
    args = parser.parse_args()

    if os.path.exists(args.model_path):
        print('\033[32;1mLoading Model...\033[0m')
        doc2vecc = Doc2VecC.load(args.model_path)
    else:
        doc2vecc = Doc2VecC(args.output_train_file, args.wordvec_file, args.docvec_file, size=args.vector_size, threads=os.cpu_count(), vocab_file=args.vocab_file, keep_generated_files=True)
        docs = utils.pickleLoad(args.docs_pkl_file)
        doc2vecc.fit(docs)
        doc2vecc.save(args.model_path)