#! /usr/bin/env bash

mkdir DATA/

# Multiprocess in shell scripts

# Corpus
mkdir DATA/corpus
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.trec.gz -P DATA/corpus/ 2> /dev/null &
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs-lookup.tsv.gz -P DATA/corpus/ 2> /dev/null &

# Train
mkdir DATA/train
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-queries.tsv.gz -P DATA/train/ 2> /dev/null &
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-top100.gz -P DATA/train/ 2> /dev/null &
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-qrels.tsv.gz -P DATA/train/ 2> /dev/null &
wget https://github.com/microsoft/TREC-2019-Deep-Learning/blob/master/utils/msmarco-doctriples.py -P DATA/train/ 2> /dev/null &

# Development
mkdir DATA/dev
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docdev-queries.tsv.gz -P DATA/dev/ 2> /dev/null &
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docdev-top100.gz -P DATA/dev/ 2> /dev/null &
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docdev-qrels.tsv.gz -P DATA/dev/ 2> /dev/null &

# Test
mkdir DATA/test
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz -P DATA/test/ 2> /dev/null &
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctest2019-top100.gz -P DATA/test/ 2> /dev/null &
wget https://trec.nist.gov/data/deep/2019qrels-docs.txt -P DATA/test/ 2> /dev/null &

sleep 100

wait

echo 'Finish Downloading'
gunzip DATA/corpus/*.gz
gunzip DATA/train/*.gz
gunzip DATA/dev/*.gz
gunzip DATA/test/*.gz
