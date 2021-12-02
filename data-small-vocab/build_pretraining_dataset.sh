#!/bin/bash

data_dir=${PWD}

corpus_path=${data_dir}/tmp/corpus.bpe
corpus_dir=${data_dir}/tmp/shards

#output_dir=pretrain_tfrecords
output_dir=pretrain_tfrecords_1000

max_seq_len=128
num_processes=4
num_out_files=1000

python build_pretraining_dataset.py \
    --corpus-path=${corpus_path} \
    --corpus-dir=${corpus_dir} \
    --vocab-file=${data_dir}/dict.ru.txt \
    --output-dir=${data_dir}/${output_dir} \
    --max-seq-length=${max_seq_len} \
    --num-processes=${num_processes} \
    --blanks-separate-docs \
    --num-out-files=${num_out_files}
