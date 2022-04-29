#!/bin/bash
echo "begin training"
python search/code/run_context.py \
    --output_dir=./saved_search_context \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=codebert \
    --tokenizer_name=roberta-base \
    --do_train \
    --data_path=context_data/split\
    --epoch 5 \
    --block_size 500 \
    --train_batch_size 20 \
    --eval_batch_size 1000\
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --gradient_accumulation_steps 1 \
