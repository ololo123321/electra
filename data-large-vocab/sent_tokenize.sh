#!/bin/bash

corpora_dir=/datadrive/data/monolingual_corpora/
output_file=./tmp/corpus.sentences
num_processes=6

python sent_tokenize.py \
    --corpora_dir ${corpora_dir} \
    --output_file ${output_file} \
    --num_processes ${num_processes}