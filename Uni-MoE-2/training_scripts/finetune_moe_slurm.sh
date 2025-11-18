#!/bin/bash
#SBATCH --job-name=uni_moe_v2_sft
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --time=1000:00:00
#SBATCH --nodelist=
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
# === 环境 ===
source ~/.bashrc
conda activate uni_moe_v2
module load CUDA/12.4
module load ffmpeg/6.1-gcc-11.4.0
module load nccl/2.23.4-cuda-12.4

LOG_DIR=path_to_your_log_directory
mkdir -p "$LOG_DIR"

# 进入项目目录
cd Uni-MoE-2

# 获取主节点信息
export NCCL_TIMEOUT_MS=300000
export PYTHONPATH=path_to_python_path
export TOKENIZERS_PARALLELISM=false
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12346
export GPUS_PER_NODE=8

echo "Job ID: $SLURM_JOB_ID"
echo "Number of nodes: $SLURM_NNODES"
echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Conda environment: $CONDA_DEFAULT_ENV"

echo "Checking GPU availability on node $SLURM_NODEID..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"N/A\"}')"

scontrol show hostnames $SLURM_JOB_NODELIST > hostfile
sed -i 's/$/ slots=8/' hostfile
echo "Generated hostfile:"
cat hostfile

deepspeed --num_gpus 8 --num_nodes $SLURM_NNODES --hostfile hostfile \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    uni_moe/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path HIT-TMG/Uni-MoE-2.0-Omni \
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
    --freeze_prefix vision_tower audio_tower image_generator task_hidden_fcs image_hidden_fcs visual_hidden_fcs speech_generator\
    --balanced_mega_batch_size 1000000 \
    --seed 3407 \