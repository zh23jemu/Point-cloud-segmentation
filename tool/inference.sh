#!/bin/bash
#
# Point Transformer V3 推理脚本
#
# ⚠️ 重要：支持两种参数格式
#
# 格式1: 命名参数（推荐，更清晰）
#   --input, -i    : 输入点云文件 (.txt 或 .npy) [必需]
#   --model, -m    : 模型权重文件 (.pth) [必需]
#   --output, -o   : 输出文件路径 [可选]
#                     默认: 与输入文件同目录，文件名添加 _pred 后缀
#                     例如: data/val/data.txt → data/val/data_pred.txt
#                     同时会生成: data/val/data_pred_labels.txt (仅标签)
#   --config, -c   : 配置文件路径 [可选，默认: config/seg/seg_pointtransformer_v3_EMA.yaml]
#
#   示例:
#     # 基本用法（自动生成输出文件名）
#     # 输入: data/val/data2_formatted_downsampled.txt
#     # 输出: data/val/data2_formatted_downsampled_pred.txt (完整数据+标签)
#     #      data/val/data2_formatted_downsampled_pred_labels.txt (仅标签)
#     bash tool/inference.sh --input data/val/data2_formatted_downsampled.txt --model exp/seg/pointtransformer_v3_EMA/model/model_best.pth
#     
#     # 指定输出文件
#     bash tool/inference.sh -i data/val/data2_formatted_downsampled.txt -m exp/seg/pointtransformer_v3_EMA/model/model_best.pth -o result/pred.txt
#     
#     # 指定配置文件
#     bash tool/inference.sh --input data/val/data2_formatted_downsampled.txt --model exp/seg/pointtransformer_v3_EMA/model/model_best.pth --config config/seg/seg_pointtransformer_v3_EMA.yaml
#
# 格式2: 位置参数（向后兼容）
#   参数1: <input_file>        - 必需，输入点云文件
#   参数2: <model_path>        - 必需，模型权重文件
#   参数3: [output_file|config] - 可选，智能识别（.yaml/.yml → config，否则 → output）
#   参数4: [config]            - 可选，配置文件路径（仅在参数3是输出文件时使用）
#
#   示例:
#     # 基本用法
#     bash tool/inference.sh data/val/data2_formatted_downsampled.txt exp/seg/pointtransformer_v3_EMA/model/model_best.pth
#     
#     # 指定输出文件
#     bash tool/inference.sh data/val/data2_formatted_downsampled.txt exp/seg/pointtransformer_v3_EMA/model/model_best.pth result/pred.txt
#
# ⚠️ 注意：不要将配置文件路径作为输出文件，否则配置文件会被覆盖！
#

export PYTHONPATH=./
eval "$(conda shell.bash hook)"
PYTHON=python

# 参数解析 - 支持两种格式：
# 1. 命名参数格式: --input file --model model.pth --output output.txt --config config.yaml
# 2. 位置参数格式: file model.pth [output.txt] [config.yaml] (向后兼容)

INPUT_FILE=""
MODEL_PATH=""
OUTPUT_FILE=""
CONFIG=""

# 检查是否使用命名参数格式（以--开头）
if [[ "$1" == --* ]]; then
    # 使用命名参数格式
    while [[ $# -gt 0 ]]; do
        case $1 in
            --input|-i)
                INPUT_FILE="$2"
                shift 2
                ;;
            --model|-m)
                MODEL_PATH="$2"
                shift 2
                ;;
            --output|-o)
                OUTPUT_FILE="$2"
                shift 2
                ;;
            --config|-c)
                CONFIG="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1"
                echo "Use --input, --model, --output, --config"
                exit 1
                ;;
        esac
    done
    
    # 设置默认配置文件
    if [ -z "$CONFIG" ]; then
        CONFIG="config/seg/seg_pointtransformer_v3_EMA.yaml"
    fi
else
    # 使用位置参数格式（向后兼容）
    INPUT_FILE=$1
    MODEL_PATH=$2
    
    # 智能识别第3个参数：如果是.yaml文件，则认为是config；否则认为是output_file
    if [ -n "$3" ]; then
        if [[ "$3" == *.yaml ]] || [[ "$3" == *.yml ]]; then
            # 第3个参数是配置文件
            CONFIG=$3
            OUTPUT_FILE=""
        else
            # 第3个参数是输出文件
            OUTPUT_FILE=$3
            CONFIG=${4:-"config/seg/seg_pointtransformer_v3_EMA.yaml"}
        fi
    else
        OUTPUT_FILE=""
        CONFIG=${4:-"config/seg/seg_pointtransformer_v3_EMA.yaml"}
    fi
fi

# 检查必需参数
if [ -z "$INPUT_FILE" ] || [ -z "$MODEL_PATH" ]; then
    echo "Usage:"
    echo "  格式1（推荐）: bash tool/inference.sh --input <input_file> --model <model_path> [--output <output_file>] [--config <config_file>]"
    echo "  格式2（兼容）: bash tool/inference.sh <input_file> <model_path> [output_file|config] [config]"
    echo ""
    echo "命名参数格式（推荐）:"
    echo "  --input, -i    [必需] 输入点云文件路径 (.txt 或 .npy)"
    echo "  --model, -m    [必需] 模型权重文件路径 (.pth)"
    echo "  --output, -o   [可选] 输出文件路径"
    echo "  --config, -c   [可选] 配置文件路径 (默认: config/seg/seg_pointtransformer_v3_EMA.yaml)"
    echo ""
    echo "位置参数格式（向后兼容）:"
    echo "  参数1          [必需] 输入点云文件路径"
    echo "  参数2          [必需] 模型权重文件路径"
    echo "  参数3          [可选] 输出文件或配置文件（.yaml/.yml → config，否则 → output）"
    echo "  参数4          [可选] 配置文件路径（仅在参数3是输出文件时使用）"
    echo ""
    echo "Examples (命名参数格式):"
    echo "  bash tool/inference.sh --input data/test/room1.txt --model exp/model/model_best.pth"
    echo "  bash tool/inference.sh -i data/test/room1.txt -m exp/model/model_best.pth -o result.txt"
    echo "  bash tool/inference.sh --input data/test/room1.txt --model exp/model/model_best.pth --config config/seg/seg_pointtransformer_v3_EMA.yaml"
    echo ""
    echo "Examples (位置参数格式):"
    echo "  bash tool/inference.sh data/test/room1.txt exp/model/model_best.pth"
    echo "  bash tool/inference.sh data/test/room1.txt exp/model/model_best.pth result/room1_pred.txt"
    echo ""
    echo "⚠️  警告：不要将配置文件路径作为输出文件，否则配置文件会被覆盖！"
    exit 1
fi

# 检查输入文件是否存在
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found: $INPUT_FILE"
    exit 1
fi

# 检查模型文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model checkpoint not found: $MODEL_PATH"
    exit 1
fi

# 检查配置文件是否存在
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

# ⚠️ 安全检查：防止配置文件被覆盖
if [ -n "$OUTPUT_FILE" ]; then
    if [[ "$OUTPUT_FILE" == *.yaml ]] || [[ "$OUTPUT_FILE" == *.yml ]]; then
        echo "⚠️  警告：输出文件路径看起来像配置文件！"
        echo "   输出文件: $OUTPUT_FILE"
        echo "   如果这是配置文件，请使用第4个参数指定配置文件，第3个参数指定输出文件"
        echo ""
        read -p "是否继续？这可能会覆盖配置文件！(y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "已取消。"
            exit 1
        fi
    fi
fi

echo "========================================"
echo "Point Transformer V3 Inference"
echo "========================================"
echo "Input file:  $INPUT_FILE"
echo "Model path:  $MODEL_PATH"
echo "Output file: ${OUTPUT_FILE:-auto (will be generated)}"
echo "Config:      $CONFIG"
echo "========================================"
echo ""

# 运行推理
# ✅ 关键修改：添加 tool/ 前缀
if [ -z "$OUTPUT_FILE" ]; then
    $PYTHON tool/inference.py \
        --config=$CONFIG \
        --input_file=$INPUT_FILE \
        --model_path=$MODEL_PATH
else
    $PYTHON tool/inference.py \
        --config=$CONFIG \
        --input_file=$INPUT_FILE \
        --model_path=$MODEL_PATH \
        --output_file=$OUTPUT_FILE
fi

echo "========================================"
echo "Inference completed!"
echo "========================================"