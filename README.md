# ViT LocVar FMNIST

这是一个基于 PyTorch 的 FashionMNIST 位置变化分类实验项目。核心思路是把原始 28x28 的 FashionMNIST 图片嵌入到更大的画布中，构造不同位置分布的数据集，再分别用 ViT、CNN 和 MLP 做分类对比，并把实验结果统一整理到 `scripts/` 和 `visualizations/` 中。

## 项目目标

这个项目主要想回答一个问题：当目标内容在图像中的位置发生变化时，ViT 和卷积网络的表现会有什么差异。

为了验证这一点，项目设计了两类数据分布：

- A：图像在画布中随机放置
- B：图像在画布中相对固定地放置

通过组合训练集和验证集分布，可以得到四组常见实验：A_A、A_B、B_A、B_B。

## 目录说明

```text
.
├── prepare_data.py        # 生成自定义 FashionMNIST 数据集
├── train.py               # 训练与验证入口
├── utils.py               # 指标统计与通用工具
├── datasets/
│   └── custom_fmnist.py   # 自定义数据集读取
├── models/
│   ├── cnn.py             # CNN 基线模型
│   ├── mlp.py             # MLP 基线模型
│   └── vit.py             # ViT 模型
├── processed_data/        # 处理后的 .pt 数据文件
├── checkpoints/           # 训练得到的权重和日志
├── scripts/               # 绘图和分析脚本
└── visualizations/        # 所有实验图像输出
```

## 环境依赖

建议使用 Python 3.9 及以上版本，并安装项目依赖：

```bash
pip install -r requirements.txt
```

如果你使用 GPU 训练，请确保 PyTorch 对应的 CUDA 环境已正确安装。

## 数据准备

项目默认使用处理后的数据文件：

- `processed_data/A_fmnist_train.pt`
- `processed_data/A_fmnist_test.pt`
- `processed_data/B_fmnist_train.pt`
- `processed_data/B_fmnist_test.pt`

如果你想从头重新生成数据，可以执行：

```bash
python prepare_data.py
```

脚本会自动完成以下工作：

1. 下载原始 FashionMNIST 数据集
2. 将图像放置到更大的画布中，生成 A / B 两类数据
3. 保存处理后的训练集和测试集到 `processed_data/`
4. 生成示例图像并保存到 `visualizations/`

## 训练实验

训练入口是 `train.py`。你可以通过参数切换模型、数据集分布和超参数。

### 常见实验组合

训练 ViT，在 A 分布上训练并在 A 分布上验证：

```bash
python train.py \
  --model-type vit \
  --train-data ./processed_data/A_fmnist_train.pt \
  --val-data ./processed_data/A_fmnist_test.pt \
  --patch-type conv \
  --save-dir ./checkpoints/vit_A_A
```

训练 CNN，在 A 分布上训练并在 A 分布上验证：

```bash
python train.py \
  --model-type cnn \
  --train-data ./processed_data/A_fmnist_train.pt \
  --val-data ./processed_data/A_fmnist_test.pt \
  --save-dir ./checkpoints/cnn_A_A
```

做跨分布测试，观察训练和验证分布不一致时的表现：

```bash
python train.py \
  --model-type vit \
  --train-data ./processed_data/A_fmnist_train.pt \
  --val-data ./processed_data/B_fmnist_test.pt \
  --patch-type linear \
  --save-dir ./checkpoints/vit_A_B
```

同理，你也可以把 `B_fmnist_train.pt` 和 `B_fmnist_test.pt` 组合起来，得到 `B_A`、`B_B` 等实验结果。

## 训练参数

`train.py` 支持以下常用参数：

- `--model-type`：选择 `vit`、`cnn` 或 `mlp`
- `--train-data`：训练集 `.pt` 文件路径
- `--val-data`：验证集 `.pt` 文件路径
- `--img-size`：输入画布大小，默认 `64`
- `--patch-size`：ViT 的 patch 大小，默认 `8`
- `--patch-type`：ViT 的 patch embedding 方式，默认 `conv`，可选 `conv` 或 `linear`
- `--embed-dim`：ViT embedding 维度，默认 `128`
- `--depth`：Transformer 层数，默认 `4`
- `--num-heads`：注意力头数，默认 `4`
- `--epochs`：训练轮数，默认 `15`
- `--batch-size`：批大小，默认 `128`
- `--learning-rate`：学习率，默认 `1e-3`
- `--weight-decay`：权重衰减，默认 `1e-4`
- `--print-freq`：训练过程中打印日志的频率，默认 `50`
- `--save-dir`：模型保存目录

## 输出结果

训练过程中会输出每个 epoch 的日志，包括：

- 训练损失
- Top-1 Accuracy
- 验证集准确率
- 训练耗时信息

当验证集表现提升时，脚本会把当前最佳模型保存到 `--save-dir` 指定的目录中。
保存文件名会自动包含模型类型、训练/验证数据前缀以及 ViT 的 `patch-type` 和 `patch-size`，方便区分不同实验配置。

## 绘图与分析

这个项目的实验图基本都放在 `scripts/` 下，每个脚本对应一类分析图，运行后会把图片写入 `visualizations/`。

常见脚本包括：

- `scripts/plot_vit_performance_comparison.py`：对比 ViT 在不同实验设置下的整体表现
- `scripts/plot_cnn_and_vit_comparison.py`：对比 CNN 和 ViT 的实验结果
- `scripts/plot_mlp_and_vit_comparison.py`：对比 MLP 和 ViT 的实验结果
- `scripts/plot_activation_comparison.py`：分析激活相关结果
- `scripts/plot_patch_feature_and_patch_size_comparison.py`：比较 patch 特征和 patch size 的影响
- `scripts/visualize_attention.py`：生成注意力可视化图

当前已经生成的图像会保存在 `visualizations/`，包括：

- `mode_A_samples.png`
- `mode_B_samples.png`
- `cnn_and_vit_comparison.png`
- `mlp_and_vit_comparison.png`
- `activation_comparison.png`
- `patch_feature_and_patch_size_comparison.png`
- `vit_performance_comparison.png`
- `attention_visualization_A_A.png`
- `attention_visualization_A_B.png`
- `attention_visualization_B_A.png`
- `attention_visualization_B_B.png`

## 数据格式

`CustomFashionMNIST` 读取的 `.pt` 文件通常包含以下字段：

- `images`：张量，形状为 `[N, 1, 64, 64]`
- `labels`：标签张量，形状为 `[N]`
- `positions`：原始图像左上角坐标，形状为 `[N, 2]`

训练时主要使用 `images` 和 `labels`，`positions` 主要用于分析和可视化。

## 结果目录

- `checkpoints/`：保存训练权重和实验输出
- `processed_data/`：保存处理后的训练/测试数据
- `visualizations/`：保存所有实验图和可视化结果

## 注意事项

- 默认画布大小是 `64x64`，因此 ViT 的 `patch-size` 需要能整除 `img-size`
- 如果你在 ViT 上切换了 `--patch-type`，建议在 README 里记录对应实验配置，避免把 `conv` 和 `linear` 的结果混在一起
- CNN 使用自适应池化，对输入大小相对更灵活
- 如果你修改了 `prepare_data.py` 中的画布设置，也要同步调整 `train.py` 中的 `--img-size`

## 快速开始

如果你想最快跑通一次完整实验，可以依次执行：

```bash
pip install -r requirements.txt
python prepare_data.py
python train.py --model-type vit --train-data ./processed_data/A_fmnist_train.pt --val-data ./processed_data/A_fmnist_test.pt --save-dir ./checkpoints/vit_A_A
```

如果你后面要复现实验图，直接运行对应的脚本，然后去 `visualizations/` 里查看输出即可。
