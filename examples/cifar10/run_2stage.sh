#!/bin/bash

set -e

name=${1:-default}
exp_pdir=${2:-exp/hardware_sim/}
PRE_EPOCHS=100
POST_EPOCHS=100
exp_dir=${exp_pdir}/${name}
ckpt_path=${exp_dir}/checkpoints/model
final_path=${exp_dir}/checkpoints/final
strategy_path=${exp_dir}/strategy.yaml
if [[ ! -d "${exp_dir}" ]]; then
    mkdir -p ${exp_dir}
fi
# stage1: training on software
python cifar10_train_hardware_finetune.py --cfg config/stage1.yaml --save-strategy ${strategy_path} --epochs ${PRE_EPOCHS} --train_dir ${ckpt_path} --log_file ${exp_dir}/stage1.log
# stage2: simulation for training on hardware
python cifar10_train_hardware_finetune.py --cfg config/stage2_hardware.yaml --scfg ${strategy_path} --batch_size 32 --epochs ${POST_EPOCHS} --weight-decay 0 --load-from ${ckpt_path} --train_dir ${final_path} --log_file ${exp_dir}/stage2.log
