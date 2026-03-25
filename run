#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_PY="${SCRIPT_DIR}/train_fozo.py"
LOG_DIR="${SCRIPT_DIR}/logs"

mkdir -p "${LOG_DIR}"

if [[ ! -f "${TRAIN_PY}" ]]; then
  echo "[Error] train_fozo.py not found: ${TRAIN_PY}" >&2
  exit 1
fi


LAUNCHER=(python)
# LAUNCHER=(torchrun --nproc_per_node 2 --master_port 29501)

# --optimizer_mode: optimizer type
#   options: FO | MeZO | ZOAdamW | ZO_Ours


COMMON_ARGS=(
  --model_path facebook/opt-1.3b
  --dataset_name narrativeqa
  --train_split train
  --validation_split validation
  --test_split test

  --max_input_length 1280
  --max_target_length 64

  --lora_r 16
  --lora_alpha 32
  --lora_dropout none
  --lora_target_modules q_proj,k_proj,v_proj,fc1,fc2

  --batch_size 2
  --accumulation_steps 2

  --scheduler_type linear
  --warmup_ratio 0.0

  --save_total_limit 10
  --max_grad_norm 1.0

  --adam_beta1 0.95
  --adam_beta2 0.999
  --adam_eps 1e-6
  --weight_decay 0.01

  --logging_steps 10
  --eval_steps 500
  --eval_max_samples 200

  --project_name OPT1.3B-LoRA-NarrativeQA-FOZO
  --use_wandb false

  --seed 42
  --data_seed 42

  --auto_resume true
  --enable_checkpoint true
  --enable_interrupt_checkpoint_only true
  --save_on_interrupt true
  --save_final_model true
  --do_final_eval true
  --save_run_summary true
  --summary_file "${SCRIPT_DIR}/outputs/train_fozo_runs/all_runs_summary.jsonl"

  --output_root "${SCRIPT_DIR}/outputs/train_fozo_runs"
)



echo "============================================================"
echo "[FO] FO_comparison"
echo "============================================================"
"${LAUNCHER[@]}" "${TRAIN_PY}" "${COMMON_ARGS[@]}" \
  --optimizer_mode FO \
  --run_name FO_comparison \
  --batch_size 2 \
  --accumulation_steps 2 \
  --max_grad_norm 1.0 \
  --train_mode step \
  --learning_rate 1e-4 \
  --scheduler_type linear \
  --lr_schedule "" \
  --eval_steps 500 \
  --eval_max_samples 200 \
  --max_steps 10000 \
  --save_steps 10000 \
  --use_wandb false \
  --save_run_summary true \
  --summary_file "${SCRIPT_DIR}/outputs/train_fozo_runs/FO_comparison_runs_summary.jsonl" \
  --enable_checkpoint true \
  --save_steps 10000 \
  --enable_interrupt_checkpoint_only true \
  --save_on_interrupt true \
  --save_final_model false \
  --auto_resume true \
  --do_final_eval true \
  --final_eval_max_samples -1 \
  --output_root "${SCRIPT_DIR}/outputs/train_fozo_runs" \
  2>&1 | tee "${LOG_DIR}/FO_comparison_$(date +%Y%m%d_%H%M%S).log"


echo "============================================================"
echo "[ZOAdamW] ZOAdamW_comparison"
echo "============================================================"
"${LAUNCHER[@]}" "${TRAIN_PY}" "${COMMON_ARGS[@]}" \
  --optimizer_mode ZOAdamW \
  --run_name ZOAdamW_comparison \
  --batch_size 2 \
  --accumulation_steps 2 \
  --max_grad_norm 1.0 \
  --train_mode step \
  --learning_rate 1e-4 \
  --scheduler_type linear \
  --eval_steps 500 \
  --eval_max_samples 200 \
  --max_steps 10000 \
  --zo_eps 1e-3 \
  --zo_samples_init 4 \
  --zo_sample_schedule "0:4" \
  --use_wandb false \
  --save_run_summary true \
  --summary_file "${SCRIPT_DIR}/outputs/train_fozo_runs/ZOAdamW_comparison_runs_summary.jsonl" \
  --enable_checkpoint true \
  --save_steps 10000 \
  --enable_interrupt_checkpoint_only true \
  --save_on_interrupt true \
  --save_final_model false \
  --auto_resume true \
  --do_final_eval true \
  --final_eval_max_samples -1 \
  --output_root "${SCRIPT_DIR}/outputs/train_fozo_runs" \
  2>&1 | tee "${LOG_DIR}/ZOAdamW_comparison_$(date +%Y%m%d_%H%M%S).log"
  
  


echo "============================================================"
echo "[MeZO] MeZO_comparison"
echo "============================================================"
"${LAUNCHER[@]}" "${TRAIN_PY}" "${COMMON_ARGS[@]}" \
  --optimizer_mode MeZO \
  --run_name MeZO_comparison \
  --batch_size 2 \
  --accumulation_steps 2 \
  --max_grad_norm 1.0 \
  --train_mode step \
  --learning_rate 1e-4 \
  --scheduler_type linear \
  --lr_schedule "" \
  --eval_steps 500 \
  --eval_max_samples 200 \
  --max_steps 10000 \
  --zo_eps 1e-3 \
  --zo_samples_init 4 \
  --zo_sample_schedule "0:4" \
  --use_wandb false \
  --save_run_summary true \
  --summary_file "${SCRIPT_DIR}/outputs/train_fozo_runs/MeZO_comparison_runs_summary.jsonl" \
  --enable_checkpoint true \
  --save_steps 10000 \
  --enable_interrupt_checkpoint_only true \
  --save_on_interrupt true \
  --save_final_model false \
  --auto_resume true \
  --do_final_eval true \
  --final_eval_max_samples -1 \
  --output_root "${SCRIPT_DIR}/outputs/train_fozo_runs" \
  2>&1 | tee "${LOG_DIR}/MeZO_comparison_$(date +%Y%m%d_%H%M%S).log"



echo "============================================================"
echo "[ZO_Ours] ZO_Ours"
echo "============================================================"
"${LAUNCHER[@]}" "${TRAIN_PY}" "${COMMON_ARGS[@]}" \
  --optimizer_mode ZO_Ours \
  --run_name ZO_Ours \
  --batch_size 2 \
  --accumulation_steps 2 \
  --max_grad_norm 1.0 \
  --train_mode step \
  --learning_rate 1e-4 \
  --scheduler_type linear \
  --lr_schedule "" \
  --eval_steps 500 \
  --eval_max_samples 200 \
  --max_steps 10000 \
  --zo_eps 1e-3 \
  --zo_samples_init 4 \
  --zo_sample_schedule "0:4" \
  --use_adaptive_eps false \
  --u_buffer_k 1 \
  --subspace_alpha 0.5 \
  --subspace_alpha_schedule "0:1.0,200:0.98,800:0.95,2000:0.92,4000:0.88,6500:0.85,8500:0.81,9500:0.79" \
  --normalize_by_sqrt_d true \
  --use_wandb false \
  --save_run_summary true \
  --summary_file "${SCRIPT_DIR}/outputs/train_fozo_runs/ZO_Ours_runs_summary.jsonl" \
  --enable_checkpoint true \
  --save_steps 10000 \
  --enable_interrupt_checkpoint_only true \
  --save_on_interrupt true \
  --save_final_model false \
  --auto_resume true \
  --do_final_eval true \
  --final_eval_max_samples -1 \
  --output_root "${SCRIPT_DIR}/outputs/train_fozo_runs" \
  2>&1 | tee "${LOG_DIR}/ZO_Ours_$(date +%Y%m%d_%H%M%S).log"


echo "All configured runs finished."
