#!/bin/bash

device=$1
exp_type=$2
sh_dir=./exp_sh

bash $sh_dir/exp-$exp_type-training.sh en zh $device llama-2-7b-hf llama