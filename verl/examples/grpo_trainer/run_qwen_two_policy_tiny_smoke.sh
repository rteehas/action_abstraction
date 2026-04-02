#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}

if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] && [[ -f /workspace/miniconda3/etc/profile.d/conda.sh ]]; then
  source /workspace/miniconda3/etc/profile.d/conda.sh
  conda activate abstraction
fi

ray stop --force >/dev/null 2>&1 || true

MODE=${MODE:-dapo}
RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}
DATA_DIR=${DATA_DIR:-${REPO_ROOT}/verl_data/two_policy_tiny_smoke}
TRAIN_INDICES=${TRAIN_INDICES:-1606,2758}
VAL_INDICES=${VAL_INDICES:-1606,2758}
ABSTRACTION_MAX_RESP_LEN=${ABSTRACTION_MAX_RESP_LEN:-1024}
SOLVER_MAX_RESP_LEN=${SOLVER_MAX_RESP_LEN:-3072}
MAX_RESP_LEN=$(( ABSTRACTION_MAX_RESP_LEN > SOLVER_MAX_RESP_LEN ? ABSTRACTION_MAX_RESP_LEN : SOLVER_MAX_RESP_LEN ))
ABSTRACTION_MODEL_PATH=${ABSTRACTION_MODEL_PATH:-${REPO_ROOT}/merged_models/qwen3_1_7b_principle_generator_ckpt1736}
SOLVER_MODEL_PATH=${SOLVER_MODEL_PATH:-${REPO_ROOT}/solver_sft_qwen_1_7b_450}
ABSTRACTION_PROMPT_TEMPLATE_PATH=${ABSTRACTION_PROMPT_TEMPLATE_PATH:-${REPO_ROOT}/prompt_templates/sft_principle_generation.txt}
SOLVER_PROMPT_TEMPLATE_PATH=${SOLVER_PROMPT_TEMPLATE_PATH:-${REPO_ROOT}/prompt_templates/hint_conditioned_problem_solving_rich_v1.txt}
RUN_ROOT=${RUN_ROOT:-/tmp/action_abstraction/two_policy_runs}
RUN_DIR=${RUN_DIR:-${RUN_ROOT}/tiny_smoke_${MODE}_${RUN_TAG}}
ROLLOUT_DIR=${ROLLOUT_DIR:-${RUN_DIR}/rollouts}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${RUN_DIR}/checkpoints}
DATA_DATALOADER_NUM_WORKERS=${DATA_DATALOADER_NUM_WORKERS:-0}
REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-1}

mkdir -p "${RUN_DIR}" "${ROLLOUT_DIR}" "${CHECKPOINT_DIR}"

if [ ! -d "${ABSTRACTION_MODEL_PATH}" ] || [ ! -f "${ABSTRACTION_MODEL_PATH}/config.json" ]; then
  python "${REPO_ROOT}/scripts/merge_lora_adapter.py" \
    --output-dir "${ABSTRACTION_MODEL_PATH}"
fi

python "${REPO_ROOT}/scripts/build_two_policy_tiny_rl_dataset.py" \
  --output-dir "${DATA_DIR}" \
  --train-indices "${TRAIN_INDICES}" \
  --val-indices "${VAL_INDICES}"

cd "${REPO_ROOT}/verl"

MAIN_MODULE=recipe.two_policy.main_two_policy_ppo
reward_manager=naive
extra_args=()
if [ "${MODE}" = "dapo" ]; then
  MAIN_MODULE=recipe.two_policy.main_two_policy_dapo
  reward_manager=dapo
  extra_args+=(algorithm.filter_groups.enable=True)
  extra_args+=(algorithm.filter_groups.metric=acc)
  extra_args+=(algorithm.filter_groups.max_num_gen_batches=2)
  extra_args+=(reward.reward_kwargs.max_resp_len="${SOLVER_MAX_RESP_LEN}")
fi

args=(
  hydra.searchpath=[file://${REPO_ROOT}/verl/verl/trainer/config]
  algorithm.adv_estimator=grpo
  data.train_files="${DATA_DIR}/train.parquet"
  data.val_files="${DATA_DIR}/val.parquet"
  data.train_batch_size=2
  data.val_batch_size=2
  data.dataloader_num_workers="${DATA_DATALOADER_NUM_WORKERS}"
  data.max_prompt_length=3072
  data.max_response_length="${MAX_RESP_LEN}"
  actor_rollout_ref.model.path="${SOLVER_MODEL_PATH}"
  actor_rollout_ref.model.lora_rank=32
  actor_rollout_ref.model.lora_alpha=32
  actor_rollout_ref.model.target_modules=all-linear
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.ppo_mini_batch_size=2
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
  actor_rollout_ref.actor.fsdp_config.param_offload=False
  +actor_rollout_ref.actor.fsdp_config.grad_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.dtype=bfloat16
  actor_rollout_ref.rollout.temperature=0.6
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45
  actor_rollout_ref.rollout.n=1
  actor_rollout_ref.rollout.response_length="${SOLVER_MAX_RESP_LEN}"
  actor_rollout_ref.rollout.enforce_eager=True
  actor_rollout_ref.rollout.free_cache_engine=True
  actor_rollout_ref.rollout.load_format=safetensors
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  abstraction_actor_rollout_ref.model.path="${ABSTRACTION_MODEL_PATH}"
  abstraction_actor_rollout_ref.model.lora_rank=32
  abstraction_actor_rollout_ref.model.lora_alpha=32
  abstraction_actor_rollout_ref.model.target_modules=all-linear
  abstraction_actor_rollout_ref.model.use_remove_padding=True
  abstraction_actor_rollout_ref.model.enable_gradient_checkpointing=True
  abstraction_actor_rollout_ref.actor.optim.lr=1e-6
  abstraction_actor_rollout_ref.actor.entropy_coeff=0
  abstraction_actor_rollout_ref.actor.ppo_mini_batch_size=2
  abstraction_actor_rollout_ref.actor.use_dynamic_bsz=True
  abstraction_actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576
  abstraction_actor_rollout_ref.actor.use_kl_loss=True
  abstraction_actor_rollout_ref.actor.kl_loss_coef=0.001
  abstraction_actor_rollout_ref.actor.kl_loss_type=low_var_kl
  abstraction_actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
  abstraction_actor_rollout_ref.actor.fsdp_config.param_offload=False
  +abstraction_actor_rollout_ref.actor.fsdp_config.grad_offload=False
  abstraction_actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
  abstraction_actor_rollout_ref.rollout.tensor_model_parallel_size=1
  abstraction_actor_rollout_ref.rollout.name=vllm
  abstraction_actor_rollout_ref.rollout.dtype=bfloat16
  abstraction_actor_rollout_ref.rollout.temperature=0.6
  abstraction_actor_rollout_ref.rollout.gpu_memory_utilization=0.45
  abstraction_actor_rollout_ref.rollout.n=1
  abstraction_actor_rollout_ref.rollout.response_length="${ABSTRACTION_MAX_RESP_LEN}"
  abstraction_actor_rollout_ref.rollout.enforce_eager=True
  abstraction_actor_rollout_ref.rollout.free_cache_engine=True
  abstraction_actor_rollout_ref.rollout.load_format=safetensors
  abstraction_actor_rollout_ref.ref.fsdp_config.param_offload=True
  two_policy.abstraction_prompt_template_path="${ABSTRACTION_PROMPT_TEMPLATE_PATH}"
  two_policy.solver_prompt_template_path="${SOLVER_PROMPT_TEMPLATE_PATH}"
  two_policy.num_abstractions=2
  two_policy.num_solver_rollouts=2
  two_policy.validation_num_abstractions=2
  two_policy.validation_num_solver_rollouts=2
  trainer.critic_warmup=0
  trainer.val_before_train=True
  trainer.logger=[console]
  trainer.project_name=verl_two_policy_smoke
  trainer.experiment_name="tiny_${MODE}_${RUN_TAG}"
  trainer.n_gpus_per_node=1
  trainer.nnodes=1
  trainer.rollout_data_dir="${ROLLOUT_DIR}"
  trainer.default_local_dir="${CHECKPOINT_DIR}"
  trainer.save_freq=1
  trainer.test_freq=1
  trainer.total_epochs=1
  trainer.resume_mode=disable
  trainer.max_actor_ckpt_to_keep=2
  reward.custom_reward_function.path="${REPO_ROOT}/verl/verl/utils/reward_score/deepscaler_math_reward_multibox_patched.py"
  reward.custom_reward_function.name=compute_score
  reward.num_workers="${REWARD_NUM_WORKERS}"
  reward.reward_manager.name="${reward_manager}"
  algorithm.use_kl_in_reward=False
  ray_kwargs.ray_init.num_cpus=32
)

python -m "${MAIN_MODULE}" "${args[@]}" "${extra_args[@]}" "$@" | tee "${RUN_DIR}/smoke.log"
