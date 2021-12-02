#!/bin/bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Clone fastBPE'
git clone https://github.com/glample/fastBPE.git

SCRIPTS=mosesdecoder/scripts
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl

data_dir=./tmp
# the same format as for bert:
# * one sentence per line
# * blank line betweeb docs
CORPUS=${data_dir}/corpus.sentences
lang=ru

tok_file=${data_dir}/corpus.tok
num_threads=6
#num_threads=1

echo "pre-processing train data..."
rm ${tok_file}
cat ${CORPUS} | \
    perl $NORM_PUNC $lang | \
    perl $REM_NON_PRINT_CHAR | \
    perl $TOKENIZER -threads ${num_threads} -a -l ${lang} >> ${tok_file}

pushd fastBPE
g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
popd

bpe_file=${data_dir}/corpus.bpe
rm ${bpe_file}
fastBPE/fast applybpe ${bpe_file} ${tok_file} bpecodes.${lang} dict.${lang}.txt
