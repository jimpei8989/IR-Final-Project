#! /usr/bin/env bash

mkdir data/

# Multiprocess in shell scripts

# Corpus
mkdir data/corpus
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.trec.gz -P data/corpus/ 2> /dev/null &
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs-lookup.tsv.gz -P data/corpus/ 2> /dev/null &

# Train
mkdir data/train
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-queries.tsv.gz -P data/train/ 2> /dev/null &
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-top100.gz -P data/train/ 2> /dev/null &
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctrain-qrels.tsv.gz -P data/train/ 2> /dev/null &
wget https://raw.githubusercontent.com/microsoft/TREC-2019-Deep-Learning/master/utils/msmarco-doctriples.py -P data/train/ 2> /dev/null &

# Development
mkdir data/dev
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docdev-queries.tsv.gz -P data/dev/ 2> /dev/null &
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docdev-top100.gz -P data/dev/ 2> /dev/null &
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docdev-qrels.tsv.gz -P data/dev/ 2> /dev/null &

# Test
mkdir data/test
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz -P data/test/ 2> /dev/null &
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-doctest2019-top100.gz -P data/test/ 2> /dev/null &
wget https://trec.nist.gov/data/deep/2019qrels-docs.txt -P data/test/ 2> /dev/null &

sleep 100

wait

echo 'Finish Downloading'
gunzip data/corpus/*.gz
gunzip data/train/*.gz
gunzip data/dev/*.gz
gunzip data/test/*.gz
