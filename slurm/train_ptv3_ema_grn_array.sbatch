#!/bin/bash
#
# PTv3 + EMA + RFG 训练脚本（Slurm array 版本）
#
# 使用方式：
#   sbatch slurm/train_ptv3_ema_grn_array.sbatch
#
# 说明：
#   1. 默认提交 2 个 array task，并允许 2 个任务并行运行，用于连续两次独立训练验证稳定性。
#   2. 不修改 YAML 中的训练参数；脚本只覆盖 save_path，避免覆盖原 EMA-only 实验目录。
#   3. 如果服务器使用 conda，请在 ENV_SETUP 中填入你的环境激活命令。
#   4. 如果服务器使用项目本地 .venv，请保持默认的 PYTHON_BIN=.venv/bin/python。

#SBATCH --job-name=ptv3_ema_rfg
#SBATCH --partition=aws
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=1-2%2
#SBATCH --output=slurm/logs/%x_%A_%a.out
#SBATCH --error=slurm/logs/%x_%A_%a.err

set -euo pipefail

# 切到 sbatch 提交时所在的项目根目录，保证相对路径和本地运行一致。
cd "${SLURM_SUBMIT_DIR}"

mkdir -p slurm/logs

# 新集群当前可用 Python 模块为 3.11.7；项目训练仍通过本地 .venv/bin/python 执行，
# module 主要用于保持编译扩展、动态库和命令行工具的基础环境一致。
module purge
module load python/3.11.7
module load gcc/11.4.0

# 新集群计算节点提供 /usr/local/cuda；L40S 对应 CUDA 架构为 8.9。
# 这里显式设置 CUDA_HOME 和编译器，保证 pointops / spconv 等 CUDA 相关依赖运行环境稳定。
export CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
export CC=$(which gcc)
export CXX=$(which g++)
export CUDAHOSTCXX=$(which g++)
export TORCH_CUDA_ARCH_LIST="8.9"

# 小数据集 + HPC 环境下，多进程 DataLoader/BLAS 线程有时会在进入首个 batch 前触发底层 FPE。
# 这里限制 CPU 数值库线程，训练命令再覆盖 workers=0，优先保证服务器可稳定复现实验。
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export PYTHONPATH=./

# 如服务器需要 conda，请取消下面两行注释，并按实际环境名修改。
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate ptv3

# 默认使用项目本地虚拟环境；若服务器环境不同，可提交时覆盖：
#   sbatch --export=ALL,PYTHON_BIN=/path/to/python slurm/train_ptv3_ema_grn_array.sbatch
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

CONFIG_PATH="config/seg/seg_pointtransformer_v3_EMA.yaml"
RUN_ID="${SLURM_ARRAY_TASK_ID:-1}"
# 实验前缀默认切到 RFG，表示当前主线为“EMA + 残差特征门控”。
# 如需复用旧 SSM/GRN 目录，可在 sbatch 时传入对应 EXP_PREFIX。
EXP_PREFIX="${EXP_PREFIX:-pointtransformer_v3_EMA_RFG}"
EXP_NAME="${EXP_PREFIX}_run${RUN_ID}"
SAVE_PATH="exp/seg/${EXP_NAME}"

mkdir -p "${SAVE_PATH}/model" "${SAVE_PATH}/result" "${SAVE_PATH}/plots"

echo "============================================================"
echo "PTv3 + EMA + RFG Slurm Training"
echo "============================================================"
echo "Host:        $(hostname)"
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Array ID:    ${RUN_ID}"
echo "Python:      ${PYTHON_BIN}"
echo "Config:      ${CONFIG_PATH}"
echo "Save path:   ${SAVE_PATH}"
echo "Submit dir:  ${SLURM_SUBMIT_DIR}"
echo "Start time:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "错误：找不到可执行 Python：${PYTHON_BIN}"
    echo "请先在服务器创建 .venv，或用 --export=ALL,PYTHON_BIN=/path/to/python 指定解释器。"
    exit 1
fi

if [ ! -f "${CONFIG_PATH}" ]; then
    echo "错误：配置文件不存在：${CONFIG_PATH}"
    exit 1
fi

"${PYTHON_BIN}" -u tool/train.py \
    --config="${CONFIG_PATH}" \
    save_path "${SAVE_PATH}" \
    workers 0

echo "============================================================"
echo "Finished time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Best model:    ${SAVE_PATH}/model/model_best.pth"
echo "Last model:    ${SAVE_PATH}/model/model_last.pth"
echo "============================================================"
