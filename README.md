# ViT Classification Based on Location-Variable FashionMNIST

基于 PyTorch 的 FashionMNIST 位置变化分类实验。这个项目把原始 28x28 的 FashionMNIST 图像放到更大的画布上，构造出两种不同分布的数据集，用 ViT 和 CNN 两种模型做分类对比。

## 项目简介

原始 FashionMNIST 图片会被放置到 64x64 的画布中，形成自定义数据集：

- Mode A: 每张图像随机放置在画布中的不同位置
- Mode B: 每张图像固定放置在画布中心附近

训练脚本支持两种模型：

- ViT: 使用 patch embedding + Transformer Encoder
- CNN: 作为卷积基线模型

## 目录结构

```text
.
├── prepare_data.py        # 生成自定义 FashionMNIST 数据集
├── train.py               # 训练与验证入口
├── utils.py               # 训练指标工具函数
├── datasets/
│   └── custom_fmnist.py   # 自定义数据集读取类
├── models/
│   ├── vit.py             # ViT 模型
│   └── cnn.py             # CNN 基线模型
├── processed_data/        # 处理后的 .pt 数据文件
├── checkpoints/           # 模型权重保存目录
└── visualizations/        # 数据可视化结果
```

## 环境依赖

建议使用 Python 3.9+，并安装 `requirements.txt` 中的依赖。

```bash
pip install -r requirements.txt
```

如果你使用的是支持 GPU 的 PyTorch 环境，请确保已正确安装对应版本的 CUDA 依赖。

## 数据准备

项目默认使用处理后的数据文件：

- `processed_data/A_fmnist_train.pt`
- `processed_data/A_fmnist_test.pt`
- `processed_data/B_fmnist_train.pt`
- `processed_data/B_fmnist_test.pt`

如果你想重新生成数据，可以直接运行：

```bash
python prepare_data.py
```

该脚本会：

1. 下载原始 FashionMNIST 数据集
2. 生成 Mode A 和 Mode B 的训练/测试集
3. 保存为 `.pt` 文件到 `processed_data/`
4. 生成若干可视化样本到 `visualizations/`

## 训练说明

训练入口是 `train.py`。你可以通过参数切换模型、数据集和超参数。

### 训练 ViT

```bash
python train.py \
	--model-type vit \
	--train-data ./processed_data/A_fmnist_train.pt \
	--val-data ./processed_data/A_fmnist_test.pt \
	--save-dir ./checkpoints/vit_A_A
```

### 训练 CNN

```bash
python train.py \
	--model-type cnn \
	--train-data ./processed_data/A_fmnist_train.pt \
	--val-data ./processed_data/A_fmnist_test.pt \
	--save-dir ./checkpoints/cnn_A_A
```

### 交叉分布评估示例

如果想测试训练集和验证集分布不一致的情况，可以这样运行：

```bash
python train.py \
	--model-type vit \
	--train-data ./processed_data/A_fmnist_train.pt \
	--val-data ./processed_data/B_fmnist_test.pt \
	--save-dir ./checkpoints/vit_A_B
```

## 训练参数

`train.py` 支持以下常用参数：

- `--model-type`: 选择 `vit` 或 `cnn`
- `--train-data`: 训练集 `.pt` 文件路径
- `--val-data`: 验证集 `.pt` 文件路径
- `--img-size`: 输入画布大小，默认 `64`
- `--patch-size`: ViT patch 大小，默认 `8`
- `--embed-dim`: ViT embedding 维度，默认 `128`
- `--depth`: Transformer 层数，默认 `4`
- `--num-heads`: 注意力头数，默认 `4`
- `--epochs`: 训练轮数，默认 `15`
- `--batch-size`: 批大小，默认 `128`
- `--learning-rate`: 学习率，默认 `1e-3`
- `--weight-decay`: 权重衰减，默认 `1e-4`
- `--save-dir`: 模型保存目录

## 输出结果

训练过程中会打印每个 epoch 的训练日志，包括：

- Batch 时间
- Loss
- Top-1 Accuracy
- 验证集准确率

当验证集准确率提升时，脚本会自动保存当前最佳模型权重到 `--save-dir` 指定目录下。

## 数据格式

`CustomFashionMNIST` 读取的 `.pt` 文件包含以下三个字段：

- `images`: 张量，形状为 `[N, 1, 64, 64]`
- `labels`: 标签张量，形状为 `[N]`
- `positions`: 原始图像左上角坐标，形状为 `[N, 2]`

训练时只使用 `images` 和 `labels`，`positions` 主要用于数据分析与可视化。

## 结果文件说明

- `checkpoints/`: 保存训练得到的模型参数
- `processed_data/`: 保存生成好的训练/测试数据
- `visualizations/`: 保存数据样本可视化图片

## 备注

- 默认输入画布大小是 `64x64`，因此 ViT 的 `patch-size` 需要保证可以整除图像大小
- CNN 使用自适应池化，因此对输入大小更灵活
- 如果你修改了 `prepare_data.py` 中的画布大小，请同步调整 `train.py` 中的 `--img-size`

## 快速开始

如果你想最快跑通一次实验，可以按下面的顺序执行：

```bash
pip install -r requirements.txt
python prepare_data.py
python train.py --model-type vit --train-data ./processed_data/A_fmnist_train.pt --val-data ./processed_data/A_fmnist_test.pt --save-dir ./checkpoints/vit_A_A
```
