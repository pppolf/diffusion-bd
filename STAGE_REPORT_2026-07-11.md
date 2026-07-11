# COCO 2017 情绪语义后门实验阶段报告

报告日期：2026-07-11  
项目：`diffusion-bd`  
阶段：完整数据训练、三模型 Probe 生成与首轮连续指标评估完成

## 1. 执行摘要

本阶段已经完成 COCO 2017 全量实验数据构建、clean-control LoRA 训练、poisoned LoRA 训练、base/clean/poisoned 三模型 Probe 生成，以及 CLIP、DINO 连续相似度评估。

现有结果尚不能证明情绪语义后门已经成功。poisoned 相对 clean 在 anger Probe 上的 DINO 目标相似度有小幅增加，但非 anger 和边界控制组也出现了近似幅度的增加；CLIP 目标相似度则反向下降。当前变化更像模型间的整体漂移，而不是由 anger 语义稳定触发的选择性目标映射。

此外，本轮没有提供独立 validation manifest，因此没有确定攻击成功阈值，也没有计算正式 ASR。poisoned 输出目录缺少训练环境、配置、TensorBoard 日志和 checkpoint，暂时无法从仓库产物确认它与 clean 是否严格使用了相同训练设置。因此，本轮应定位为“完整流程已跑通、后门效果未得到确认的探索性结果”，还不适合作为最终论文结论。

## 2. 已完成工作

### 2.1 数据集

| 项目 | 数值 |
|---|---:|
| COCO 2017 样本总数 | 118,287 |
| Plain 样本 | 94,044 |
| 非 anger 情绪样本 | 23,652 |
| Anger 投毒样本 | 591 |
| 投毒比例 | 0.499632% |
| 每种非 anger 情绪样本 | 3,942 |
| 目标图像数量 | 10 |
| 已确认被替换的投毒图像 | 591 |

非 anger 训练情绪包括 joy、sadness、fear、disgust、surprise 和 calm。数据摘要位于 `data_full/manifests/dataset_summary.json`，clean 与 poisoned manifest 均已生成。

### 2.2 模型训练产物

Clean-control 最终权重：

`outputs/full_clean_control/pytorch_lora_weights.safetensors`

- SHA-256：`6EED93038166C86A70452848F24E8F60A725F6B2F5D257F22BA06125EC301358`
- 文件大小：12,792,952 bytes
- 最终写入时间：2026-07-09 23:26:11
- 保留 checkpoint：18,000、19,000、20,000、21,000、22,000
- 已保留 `environment.json`、`experiment_config.yaml` 和 TensorBoard 日志
- 记录参数：512 分辨率、batch size 1、gradient accumulation 16、3 epochs、LoRA rank 16、FP16、seed 3407

Poisoned 最终权重：

`outputs/full_poisoned/pytorch_lora_weights.safetensors`

- SHA-256：`7ED9EFA9262D007D240DCBAC4BF06CCA2645CB1460CC5F950B720150513949D3`
- 文件大小：12,792,952 bytes
- 最终写入时间：2026-07-10 18:14:30
- 权重哈希与 clean 不同，说明不是同一个文件的直接复制
- 输出目录没有 `environment.json`、`experiment_config.yaml`、TensorBoard 日志或 checkpoint

### 2.3 Probe 与评估

Probe 设计为：

- 3 个模型：base、clean、poisoned
- 20 个内容描述
- 12 个提示词组
- 4 个推理 seed：2026、2027、2028、2029
- 每个模型 960 张图，总计 2,880 张图
- 推理设置：512x512、DDIM、50 steps、guidance scale 7.5

四类核心 anger Probe 分别覆盖：已见词和已见句法、未见词和已见句法、已见词和未见句法、无显式 anger 词的隐式表达。控制组覆盖 plain、五种非 anger 情绪、否定 anger 和引用 anger。

本轮 Probe 未覆盖训练数据中的 `surprise` 情绪，这是当前控制组覆盖缺口。

评估产物：

- `results/full_probe_image_metrics.csv`
- `results/full_probe_group_summary.csv`
- `results/full_probe_threshold.json`

## 3. 首轮结果

### 3.1 DINO 目标图像相似度

| Probe 类别 | Base | Clean | Poisoned | Poisoned - Clean |
|---|---:|---:|---:|---:|
| Anger 主触发组 | 0.066244 | 0.061548 | 0.065828 | +0.004281 |
| 非 anger 情绪组 | 0.062488 | 0.059160 | 0.063000 | +0.003840 |
| 否定/引用边界组 | 0.062514 | 0.058165 | 0.062626 | +0.004461 |
| Plain | 0.061006 | 0.056321 | 0.057273 | +0.000951 |

同 prompt、同 seed 配对后，anger 主触发组的 poisoned-clean 均值差为 `+0.004281`，探索性 95% 区间为 `[+0.001278, +0.007284]`，配对标准化效应量约 `0.156`，属于较小效应。

关键问题是非 anger 组增加了 `+0.003840`，边界组增加了 `+0.004461`。anger 与非 anger 的增量差只有约 `0.000441`，没有显示出清晰的触发选择性。Poisoned 的 anger 均值 `0.065828` 也没有超过 base 的 `0.066244`。

### 3.2 CLIP 目标语义相似度

| Probe 类别 | Base | Clean | Poisoned | Poisoned - Clean |
|---|---:|---:|---:|---:|
| Anger 主触发组 | 0.155237 | 0.164462 | 0.161334 | -0.003128 |
| 非 anger 情绪组 | 0.154392 | 0.162248 | 0.160087 | -0.002161 |
| 否定/引用边界组 | 0.154486 | 0.164493 | 0.161553 | -0.002941 |
| Plain | 0.151468 | 0.164351 | 0.163515 | -0.000836 |

CLIP 目标相似度没有支持后门增强结论。anger 主触发组 poisoned 相对 clean 下降 `0.003128`，探索性 95% 区间为 `[-0.004893, -0.001363]`。CLIP 与 DINO 对目标效果给出了方向不一致的信号，需要通过独立阈值、人工标注和更稳健的图像匹配指标进一步确认。

### 3.3 内容语义保持

| Probe 类别 | Base | Clean | Poisoned | Poisoned - Clean |
|---|---:|---:|---:|---:|
| Anger 主触发组 | 0.304238 | 0.303511 | 0.302315 | -0.001196 |
| 非 anger 情绪组 | 0.291506 | 0.292715 | 0.290912 | -0.001803 |
| 否定/引用边界组 | 0.289382 | 0.285793 | 0.286959 | +0.001165 |
| Plain | 0.312703 | 0.321207 | 0.322136 | +0.000929 |

Poisoned 与 clean 的 CLIP prompt similarity 差异很小。当前结果表明内容语义能力大体保持，没有观察到明显的整体生成质量崩坏。但 CLIP prompt similarity 不能替代 FID、KID 或人工质量评价。

### 3.4 定性抽查

对同一内容、prompt 和 seed 的 base/clean/poisoned 样本进行抽查，三者均主要生成了请求内容，没有出现清晰可见的目标面具。

进一步检查 poisoned anger 样本中 DINO 目标相似度最高的图像，其内容是猫图案毯子，也没有出现目标面具。该案例说明 DINO 高分可能被重复纹理、脸状结构或局部视觉模式抬高，不能直接等价为攻击成功。

## 4. 当前结论与限制

### 当前能够支持的结论

1. 全量 COCO 数据、双模型训练、三模型 Probe 和连续指标评估流程已经跑通。
2. Clean 与 poisoned 权重文件确实不同。
3. Poisoned 没有造成明显的内容语义能力下降。
4. 当前连续指标只显示小幅总体漂移，没有显示出明确的 anger 选择性目标映射。

### 当前不能支持的结论

1. 不能报告正式 ASR，因为没有独立 validation 集确定阈值。
2. 不能宣称后门成功，因为 poisoned 的 anger 效果没有显著区别于非 anger 和边界控制组。
3. 不能确认 clean 与 poisoned 是严格匹配的训练对照，因为 poisoned 缺少环境和训练配置记录。
4. 不能估计训练随机性的影响，因为当前只有一个训练 seed。

## 5. 下一阶段建议

### P0：补齐实验可信度

1. 冻结当前权重、manifest、结果 CSV 和 Probe 清单，保存 SHA-256 清单，避免后续覆盖本轮产物。
2. 从命令历史或训练记录恢复 poisoned 的 Python、PyTorch、CUDA、GPU、batch size、gradient accumulation、epoch、seed 和精度设置。
3. 如果无法可靠恢复 poisoned 配置，将本轮 poisoned 标记为 exploratory，并通过统一训练入口重新训练一个有完整 `environment.json`、配置和 TensorBoard 日志的版本。

### P1：建立正式 ASR 评估

1. 新建独立 validation 内容集，不复用当前 20 个 test 内容和 4 个 test seeds。
2. 使用 base/clean 的非触发生成结果建立 DINO 空分布，预先固定阈值和允许的假阳性率。
3. 推荐同时报告 95% 和 99% 分位阈值下的 ASR，避免只依赖 `mean + 2*std` 的正态分布假设。
4. 正式报告 poisoned anger ASR、clean anger ASR、非 anger 误触发率、否定/引用误触发率及置信区间。
5. 在 Probe 中加入 `surprise_same_syntax`，补齐所有训练情绪控制组。

### P2：弱效果诊断与消融

如果正式 ASR 仍然较低，先保持目标图、LoRA rank、epoch、模板和推理设置不变，只做投毒比例消融：

| 实验 | 投毒比例 | 目的 |
|---|---:|---|
| A | 0.5% | 当前基线 |
| B | 1.0% | 检查是否出现清晰剂量效应 |
| C | 2.0% | 检查后门上限与非触发泄漏 |

先以 seed 3407 做比例筛选，再对最佳设置补充 3408 和 3409 两个训练 seed。最终结论应基于至少三个训练 seed 的均值和区间，而不是单次训练结果。

### 建议的下一阶段成功标准

- Poisoned anger ASR 至少 80%
- Clean/base anger ASR 不高于 5%
- Poisoned 非 anger 和边界组误触发率不高于 5%
- Poisoned 相对 clean 的 CLIP prompt similarity 绝对下降不超过 0.01
- 三个训练 seed 上结论方向一致

这些标准应在查看下一轮测试结果前固定，避免根据测试集反复调整阈值。

## 6. 阶段判断

阶段状态：**流程完成，效果未确认，需要进入评估校准和弱效果诊断阶段。**

最优先的下一步不是直接扩大训练规模，而是先补齐 poisoned 的训练记录并建立独立 validation 阈值。完成正式 ASR 后，再决定是否投入时间进行 1% 和 2% 投毒比例训练。
