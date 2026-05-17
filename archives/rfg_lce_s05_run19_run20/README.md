# RFG_LCE_S05 run19/run20 独立归档

本目录单独保存 `RFG_LCE_S05` 的配置快照与 run19/run20 实验结果，避免后续主线切换到 `RFG_LCE_S06` 后丢失 S05 当时的实验语境。

## 归档内容

- `config_snapshot/`：从提交 `c120371` 取出的 S05 代码与配置快照。
  - `model_pointtransformer_v3_S05.py`：S05 当时的模型实现，`lce_fusion_weight=0.04`。
  - `seg_pointtransformer_v3_EMA_S05.yaml`：训练配置快照。
  - `train_ptv3_ema_rfg_lce_s05_array.sbatch`：S05 训练脚本快照，默认 `EXP_PREFIX=pointtransformer_v3_EMA_RFG_LCE_S05`。
  - `infer_results_files_s05.sbatch`：S05 固定四文件推理脚本快照。
- `exp/seg/pointtransformer_v3_EMA_RFG_LCE_S05_run19/`：run19 的 best_results、训练曲线和固定四文件推理日志。
- `exp/seg/pointtransformer_v3_EMA_RFG_LCE_S05_run20/`：run20 的 best_results、训练曲线和固定四文件推理日志。
- `slurm/logs/`：run19/run20 训练日志以及对应固定四文件推理日志。
- `manifest.json`：归档来源、文件清单和作业号记录。

## 实验来源

- 方案：PTv3 + EMA + 限幅 RFG + LCE
- 训练作业号：`33274442`
- 推理作业号：`33280740`、`33280741`
- 配置快照来源提交：`c120371`
- 固定四文件目标线来自旧 EMA-only +1%：
  - `data2`：0.9735
  - `shanqu6`：0.9674
  - `shanqu2`：0.9675
  - `data11`：0.9707

## 关键结果

| 文件 | run19 class1 IoU | run20 class1 IoU | 是否连续达标 |
|---|---:|---:|---|
| `data2` | 0.9585 | 0.9570 | 否 |
| `shanqu6` | 0.9726 | 0.9724 | 是 |
| `shanqu2` | 0.9799 | 0.9801 | 是 |
| `data11` | 0.9779 | 0.9782 | 是 |

训练验证集：

| run | Best Epoch | Validation mIoU |
|---|---:|---:|
| 19 | 49 | 0.9709 |
| 20 | 49 | 0.9690 |

## 结论

S05 证明 LCE 对 `shanqu6`、`shanqu2`、`data11` 的 class1 IoU 有明显帮助，三份固定文件连续超过旧 EMA-only +1% 目标线。但 `data2` 连续低于目标线且低于旧基线，因此 S05 不能作为最终达标方案。

后续 S06 的设计动机就是在 S05 基础上降低 LCE 融合权重，并加入 LCE 残差软限幅，优先修复 `data2` 失守问题。
