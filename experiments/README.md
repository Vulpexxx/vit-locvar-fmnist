# 实验管理总说明

这个目录专门放实验编排、记录和结果汇总，不放模型源码。

## 目录约定

- `00_template/`：新实验复制用的模板
- `01_baselines/`：基础对比实验，例如 ViT 在 A/A、B/B 数据上的训练
- `02_cross_domain/`：交叉分布实验，例如 A/B、B/A
- `03_comparison/`：和 CNN、MLP 的对比实验
- `04_ablation/`：消融实验，例如 patch size、激活函数、RoPE、patch 表示方式

## 每个实验目录建议保存的内容

- `config.yaml`：这次实验的参数
- `notes.md`：实验目的、现象、结论
- `metrics.csv`：每个 epoch 的训练和验证指标
- `result_summary.md`：最终总结
- `run.txt`：启动命令和运行备注
- `plots/`：曲线图、注意力图、混淆矩阵
- `artifacts/`：如果需要，放这个实验专属权重或中间文件

## 命名建议

- 格式建议：`exp_编号_模型_训练集_测试集_关键参数`
- 示例：`exp_001_vit_AA_patch4`
- 如果你后面实验很多，编号优先，便于排序和查找
