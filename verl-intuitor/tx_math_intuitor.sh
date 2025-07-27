#!/bin/bash
#SBATCH --job-name=math_intuitor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --partition=accelerated
#SBATCH --time=6:00:00
#SBATCH --output=/hkfs/work/workspace/scratch/hgf_sap9939-myspace/intuitor/logs/math_intuitor_%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/hgf_sap9939-myspace/intuitor/logs/math_intuitor_%j.err

set -x
unset ROCR_VISIBLE_DEVICES

export WANDB_API_KEY=7526e35bcef1a5880516f1a32352cd3e6c5a4d8a
export ACCELERATE_LOG_LEVEL=info
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

source ~/.bashrc
conda activate pure

# 创建日志和 checkpoint 目录
mkdir -p /hkfs/work/workspace/scratch/hgf_sap9939-myspace/intuitor/logs
mkdir -p /hkfs/work/workspace/scratch/hgf_sap9939-myspace/intuitor/checkpoints

srun python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=intuitor \
    data.train_files=$HOME/data/math/train.parquet \
    data.val_files=$HOME/data/math/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.005 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=${GPUS_PER_NODE:-4} \
    trainer.nnodes=${NODES:-1} \
    trainer.logger=['console','wandb'] \
    trainer.project_name=verl \
    trainer.experiment_name=math_intuitor \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    trainer.default_local_dir=/hkfs/work/workspace/scratch/hgf_sap9939-myspace/intuitor/checkpoints \
    2>&1 | tee /hkfs/work/workspace/scratch/hgf_sap9939-myspace/intuitor/logs/verl_math_intuitor.log