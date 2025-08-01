!/bin/bash
#SBATCH --job-name=verl_math_intuitor
#SBATCH --output=verl_math_intuitor.%j.out
#SBATCH --error=verl_math_intuitor.%j.err
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=3:00:00

set -x
source ~/.bashrc
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
echo "ROCR_VISIBLE_DEVICES='$ROCR_VISIBLE_DEVICES'"
echo "HIP_VISIBLE_DEVICES='$HIP_VISIBLE_DEVICES'"
echo "CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'"

# 设置工作空间目录
WORKSPACE_DIR="/hkfs/work/workspace/scratch/hgf_teb8892-zt_space"
export WORKSPACE_DIR="/hkfs/work/workspace/scratch/hgf_teb8892-zt_space"
mkdir -p "${WORKSPACE_DIR}"

ls -ld /hkfs/work/workspace/scratch/hgf_teb8892-zt_space

chmod u+rwx /hkfs/work/workspace/scratch/hgf_teb8892-zt_space


export ACCELERATE_LOG_LEVEL=info
export HYDRA_FULL_ERROR=1
export WANDB_DIR=${WORKSPACE_DIR}/wandb_logs


cd /home/hk-project-p0022560/hgf_teb8892/projects/Intuitor/verl-intuitor
conda activate pure

# 创建必要的目录
mkdir -p ${WANDB_DIR}
mkdir -p ${WORKSPACE_DIR}/checkpoints
mkdir -p ${WORKSPACE_DIR}/logs

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
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
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=verl \
    trainer.experiment_name=math_intuitor \
    +trainer.save_dir=${WORKSPACE_DIR}/checkpoints \
    trainer.test_freq=10 \
    trainer.save_freq=10  \
    trainer.total_epochs=1 2>&1 | tee ${WORKSPACE_DIR}/verl_math_intuitor.log