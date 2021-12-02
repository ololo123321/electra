#!/bin/bash

num_gpus=2
data_dir=${PWD}/data-large-vocab

models_dir=${PWD}/models
#models_dir=${PWD}/models_wo_lr_scaling

hparams_dir=${PWD}/hparams

#model_name=small
model_name=base
#model_name=large

model_dir=${models_dir}/${model_name}
log_file=${model_dir}/log.txt
rm -r ${model_dir}
mkdir -p ${model_dir}

hparam_name="${model_name}.json"

# с чекпоинтом
#init_checkpoint=${PWD}/checkpoints/large/model.ckpt-20000
#mpiexec --allow-run-as-root -np ${num_gpus} --bind-to socket python run_pretraining.py \
#    --data-dir ${data_dir} \
#    --model-dir ${models_dir} \
#    --model-name ${model_name} \
#    --init-checkpoint ${init_checkpoint} \
#    --hparams ${hparams_dir}/${hparam_name} |& tee -a ${log_file}

# без чекпоинта
mpiexec --allow-run-as-root -np ${num_gpus} --bind-to socket python run_pretraining.py \
    --data-dir ${data_dir} \
    --model-dir ${models_dir} \
    --model-name ${model_name} \
    --hparams ${hparams_dir}/${hparam_name} |& tee -a ${log_file}