#!/bin/bash

# sh var
src=$1
tgt=$2
device=$3
fmn=$4
fmn_type=$5

# fixed path
model_type=context_training
dev_set=valid
test_set=test

# dir
CODES=./
data_dir=$CODES/data_example/$model_type/$src-$tgt
CKPT=$CODES/llms_ckpt/$fmn
python=$your_python_path

# running workspace
model_save_dir=$CODES/workspace/cmtpt-model
mkdir -p $model_save_dir

# automatically get the gpus ranks
array=(${device//,/ })
world_size=($(echo ${#array[*]}))
((rk=$world_size-1))
ranks=($(seq 0 1 $rk))
ranks=`echo "${ranks[@]}"` # | sed 's/[ ][ ]*/,/g'`
src_langs=(es zh fr de ru)
# contronl the stage of experiment
train=1
eval=1
fmn_type
# training commands
if [ $train -eq 1 ]; then
CUDA_VISIBLE_DEVICES=$device deepspeed --master_port 29519 \
 $CODES/train.py \
    --num_train_epochs 4 \
    --train_path $data_dir/train.pre_ctx.src.$src \
    --dev_path $data_dir/valid.pre_ctx.src.$src \
    --train_batch_size 256 \
    --gradient_accumulation_steps 8 \
    --model_dir $CKPT \
    --output_dir $model_save_dir \
    --log_steps 50 \
    --save_steps 2500 \
    --max_tgt_len 100 \
    --max_src_len 100 \
    --warmup_ratio 0.1 \
    --re_encoding 0 \
    --keys_type_num 0 \
    --share_prompt \
    --prompts_proj \
    --use_ctx \
    --mode $fmn_type \
    --recover_training_from $model_save_dir \
    --ctx_sent_num 3 \
    --learning_rate 5e-5 \
    --ds_file $CODES/ds_configs/zero_stage2_no_offload.json \
    --prompt_length 64 2>&1 | tee $model_save_dir/train.log
fi

# inference command
if [ $eval -eq 1 ]; then
start=1
end=20
step=1
dev_best_bleu=0.0001
dev_best_comet=0.0001
best_bleu_checkpoint=1
best_comet_checkpoint=1

# evaluation on dev set
while [ $start -le $end ]; do

if [ -f "$model_save_dir/model-$start.pt" ] && [ ! -f "$model_save_dir/$start/$dev_set.$src-$tgt.tran" ]; then
    echo "Decoding using $model_save_dir/model-$start.pt"
    mkdir -p $model_save_dir/$start
    touch $model_save_dir/$start/$dev_set.$src-$tgt.tran
    CUDA_VISIBLE_DEVICES=$device \
        $python $CODES/inference.py \
                --input $data_dir/$dev_set.pre_ctx.src.$src \
                --ptm $CKPT \
                --batch_size 1 \
                --keys_type_num 0 \
                --re_encoding 0 \
                --keys_type_num 0 \
                --share_prompt \
                --prompts_proj \
                --use_ctx \
                --mode $fmn_type \
                --output $model_save_dir/$start/$dev_set.$src-$tgt.tran \
                --half --prefix $model_save_dir/model-$start.pt \
                --device_list $ranks --decode_alpha 0.0 --prompt_length 64 2>&1 | tee $model_save_dir/$start/$dev_set.log
    $python -m sacrebleu.sacrebleu $data_dir/$dev_set.sent.$tgt -i $model_save_dir/$start/$dev_set.$src-$tgt.tran -w 2 --score-only > $model_save_dir/$start/$dev_set.$src-$tgt.tran.bleures
    comet-score -s $data_dir/$dev_set.sent.$src -t $model_save_dir/$start/$dev_set.$src-$tgt.tran -r $data_dir/$dev_set.sent.$tgt --quiet --only_system > $model_save_dir/$start/$dev_set.$src-$tgt.tran.comet
    
    BLEU_s=$(cat $model_save_dir/$start/$dev_set.$src-$tgt.tran.bleures)
    COMET_s=$(cat $model_save_dir/$start/$dev_set.$src-$tgt.tran.comet | grep -oP '\d*\.\d+')
    
    if [ `echo "$BLEU_s > $dev_best_bleu" | bc` -eq 1 ]; then
        dev_best_bleu=$BLEU_s
        best_bleu_checkpoint=$start
    fi

    if [ `echo "$COMET_s > $dev_best_comet" | bc` -eq 1 ]; then
        dev_best_comet=$COMET_s
        best_comet_checkpoint=$start
    fi
fi
start=$((${start}+$step))
done

# performance evaluation on best dev sacrebleu model
CUDA_VISIBLE_DEVICES=$device \
$python $CODES/inference.py \
        --input $data_dir/$test_set.pre_ctx.src.$src \
        --ptm $CKPT \
        --batch_size 1 \
        --keys_type_num 0 \
        --re_encoding 0 \
        --keys_type_num 0 \
        --share_prompt \
        --prompts_proj \
        --use_ctx \
        --mode $fmn_type \
        --output $model_save_dir/$start/$test_set.$src-$tgt.tran \
        --half --prefix $model_save_dir/model-$start.pt \
        --device_list $ranks --decode_alpha 0.0 --prompt_length 64 2>&1 | tee $model_save_dir/$start/$test_set.log
$python -m sacrebleu.sacrebleu $data_dir/$test_set.sent.$tgt -i $model_save_dir/$start/$test_set.$src-$tgt.tran -w 2 --score-only > $model_save_dir/$start/$test_set.$src-$tgt.tran.bleures
comet-score -s $data_dir/$test_set.sent.$src -t $model_save_dir/$start/$test_set.$src-$tgt.tran -r $data_dir/$test_set.sent.$tgt --quiet --only_system > $model_save_dir/$start/$test_set.$src-$tgt.tran.comet

BLEU_s=$(cat $model_save_dir/$start/$test_set.$src-$tgt.tran.bleures)
COMET_s=$(cat $model_save_dir/$start/$test_set.$src-$tgt.tran.comet | grep -oP '\d*\.\d+')
fi

# save the final evaluation results
echo -e "$dev_set:\n best_bleu_checkpoint: $best_bleu_checkpoint\n best_comet_checkpoint: $best_bleu_checkpoint" >> $model_save_dir/results.evalmb
