export HF_HUB_ENABLE_HF_TRANSFER="1"
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=eval/lmms-eval
export HF_TOKEN="your_hf_token"
export DECORD_EOF_RETRY_MAX=40960

python -m lmms_eval \
    --model uni_moe_2_omni \
    --model_args pretrained=HIT-TMG/Uni-MoE-2.0-Omni \
    --tasks ai2d \
    --batch_size 1 \
    --output_path eval/eval_results \
    --log_samples