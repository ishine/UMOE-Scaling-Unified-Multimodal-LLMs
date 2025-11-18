#!/bin/bash

LOG_DIR=path_to_your_log_directory
mkdir -p $LOG_DIR

# 进入项目目录
cd Uni-MoE-2

# 获取主节点信息
export PYTHONPATH=path_to_python_path
export TOKENIZERS_PARALLELISM=false
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=12346

echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Conda environment: $CONDA_DEFAULT_ENV"

deepspeed --include="localhost:0" --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    uni_moe/train/train.py \
    --deepspeed training_scripts/zero2.json \
    --model_name_or_path HIT/Uni-MoE-2.0-Omni \
    --data_path demo_100.json \
    --ep_size 1 \
    --token_drop True \
    --audio_folder path_to_audio_folder \
    --image_folder path_to_image_folder \
    --video_folder path_to_video_folder \
    --bf16 True \
    --tf32 True \
    --run_name Uni_MoE_2_SFT \
    --output_dir path_to_save_ckpt \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --aux_balance_weight 10 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 15000 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --report_to tensorboard \
    --group_by_modality_length True \
    --frames_upbound 48 \
    --attn_implementation "flash_attention_2" \
    --stable_mode True \
    --freeze_prefix vision_tower audio_tower image_generator task_hidden_fcs image_hidden_fcs visual_hidden_fcs speech_generator audio_aligner lm_head\
    --balanced_mega_batch_size 1000000 \
    --seed 3407 \
    --lora_enable True