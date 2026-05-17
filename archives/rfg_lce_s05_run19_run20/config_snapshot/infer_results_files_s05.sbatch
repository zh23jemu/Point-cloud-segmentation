#!/bin/bash
#
# results.md 中 4 个单文件推理的 Slurm 脚本
#
# 使用方式：
#   sbatch --export=ALL,RUN_ID=9 slurm/infer_results_files.sbatch
#   sbatch --export=ALL,RUN_ID=10 slurm/infer_results_files.sbatch
#
# 可选覆盖：
#   MODEL_PATH=/path/to/model_best.pth
#   PYTHON_BIN=/path/to/python

#SBATCH --job-name=ptv3_rfg_infer
#SBATCH --partition=aws
#SBATCH --account=gpo-ifv7xx
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p slurm/logs

# 四文件推理默认使用新 Slurm 集群，和后续训练任务保持在同一套路径与资源环境中。
# 项目仍通过本地 .venv/bin/python 执行，module 只负责提供基础 ABI 和命令行环境。
module purge
module load python/3.11.7
module load gcc/11.4.0

# 新集群计算节点提供 /usr/local/cuda；已验证 aws 分区节点为 NVIDIA L40S，CUDA 架构为 8.9。
# 这里显式设置 CUDA_HOME 和编译器，保证 pointops / spconv 等 CUDA 相关依赖运行环境稳定。
export CUDA_HOME=/usr/local/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
export CC=$(which gcc)
export CXX=$(which g++)
export CUDAHOSTCXX=$(which g++)
export TORCH_CUDA_ARCH_LIST="8.9"

# 与训练脚本保持一致，限制 CPU 数值库线程，避免小规模推理时被底层线程环境影响。
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export PYTHONPATH=./

# 如服务器需要 conda，请取消下面两行注释，并按实际环境名修改。
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate ptv3

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
CONFIG_PATH="config/seg/seg_pointtransformer_v3_EMA.yaml"
RUN_ID="${RUN_ID:-1}"
# 默认读取当前新实验前缀；如果要推理旧 RFG/SSM/GRN 结果，可传入对应 EXP_PREFIX。
EXP_PREFIX="${EXP_PREFIX:-pointtransformer_v3_EMA_RFG_LCE_S05}"
DEFAULT_MODEL_PATH="exp/seg/${EXP_PREFIX}_run${RUN_ID}/model/model_best.pth"
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
OUT_DIR="exp/seg/${EXP_PREFIX}_run${RUN_ID}/inference_results"

mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "PTv3 + EMA Single-file Inference"
echo "============================================================"
echo "Host:       $(hostname)"
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Python:     ${PYTHON_BIN}"
echo "Config:     ${CONFIG_PATH}"
echo "Model:      ${MODEL_PATH}"
echo "Output dir: ${OUT_DIR}"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "错误：找不到可执行 Python：${PYTHON_BIN}"
    exit 1
fi

if [ ! -f "${MODEL_PATH}" ]; then
    echo "错误：模型权重不存在：${MODEL_PATH}"
    exit 1
fi

declare -a INPUT_FILES=(
    "data/val/data2_formatted_downsampled (1).txt"
    "data/test/shanqu6_final (1).txt"
    "data/test/shanqu2_final (1).txt"
    "data/test/data11_formatted_downsampled (1).txt"
)

for INPUT_FILE in "${INPUT_FILES[@]}"; do
    if [ ! -f "${INPUT_FILE}" ]; then
        echo "错误：输入文件不存在：${INPUT_FILE}"
        exit 1
    fi

    BASE_NAME="$(basename "${INPUT_FILE}" .txt)"
    OUTPUT_FILE="${OUT_DIR}/${BASE_NAME}_pred.txt"
    LOG_FILE="${OUT_DIR}/${BASE_NAME}_infer.log"

    echo "------------------------------------------------------------"
    echo "Input:  ${INPUT_FILE}"
    echo "Output: ${OUTPUT_FILE}"
    echo "Log:    ${LOG_FILE}"

    "${PYTHON_BIN}" -u tool/inference.py \
        --config="${CONFIG_PATH}" \
        --input_file="${INPUT_FILE}" \
        --model_path="${MODEL_PATH}" \
        --output_file="${OUTPUT_FILE}" \
        --ensemble_runs 3 \
        2>&1 | tee "${LOG_FILE}"
done

echo "============================================================"
echo "Finished time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
