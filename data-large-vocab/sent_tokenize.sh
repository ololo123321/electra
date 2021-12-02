#!/bin/bash

# shards of corpus. each file: one document per line
corpora_dir=/path/to/corpora_dir
output_file=./tmp/corpus.sentences
num_processes=6

python sent_tokenize.py \
    --corpora_dir ${corpora_dir} \
    --output_file ${output_file} \
    --num_processes ${num_processes}