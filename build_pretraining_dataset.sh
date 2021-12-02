#!/bin/bash

data_dir=${PWD}/data-10m-large-vocab-clean

corpus_path=${data_dir}/tmp/corpus.sentences
corpus_dir=${data_dir}/tmp/sharded

max_seq_len=128

num_processes=12
num_out_files=1000

python build_pretraining_dataset.py \
    --corpus-path=${corpus_path} \
    --corpus-dir=${corpus_dir} \
    --vocab-file=${data_dir}/vocab.txt \
    --output-dir=${data_dir}/pretrain_tfrecords \
    --max-seq-length=${max_seq_len} \
    --num-processes=${num_processes} \
    --blanks-separate-docs \
    --num-out-files=${num_out_files}
