# MGBR: A Multi-Interaction Graph Attention Network for Bundle Recommendation

本项目为论文 "A Multi-Interaction Graph Attention Network for Bundle Recommendation" 的官方 PyTorch 代码实现，旨在为学术研究者和开发者提供可复现的捆绑推荐（Bundle Recommendation）基线与创新方法。

## 1. 项目简介

MGBR（Multi-Interaction Graph Attention Network for Bundle Recommendation）提出了一种多交互图注意力网络，通过联合建模用户-捆绑、用户-物品、捆绑-物品三类异构关系，利用多层图注意力机制和结构感知正则，有效提升了捆绑推荐任务的性能。

## 2. 方法原理

- **多图联合建模**：分别构建用户-捆绑、用户-物品、捆绑-物品三种二分图，捕捉多粒度的交互关系。
- **多层图注意力机制**：采用多种图神经网络层，融合不同视角的节点特征。
- **结构感知正则**：引入节点度数与邻居度数之和的结构损失，提升模型泛化能力。
- **BPR损失**：优化排序目标，提升推荐准确率。

## 3. 依赖环境

- OS: Ubuntu 18.04 或更高版本 / Windows 10
- Python >= 3.10.14
- PyTorch >= 2.1.1
- CUDA 11.8（可选，推荐使用GPU加速）
- 其他依赖（如 DGL, numpy, scipy, tensorboardX 等）

## 4. 数据集准备

请将原始数据集（如 Youshu）放置于 `./data` 目录下，目录结构如下：

```
data/
  └── Youshu/
        ├── Youshu_data_size.txt
        ├── user_bundle_train.txt
        ├── user_bundle_test.txt
        ├── user_item.txt
        └── bundle_item.txt
```

- `*_data_size.txt` 文件格式：`num_users \t num_bundles \t num_items`
- 其余文件为三元组或二元组交互数据，具体格式详见 `dataset.py`。

## 5. 快速开始

### 5.1 安装依赖



### 5.2 训练模型

```bash
python main.py
```

- 可通过修改 `config.py` 配置数据集、模型参数、GPU编号等。
- 支持命令行参数指定GPU和数据集。

### 5.3 测试与评估

训练过程中自动在验证集/测试集上评估 Recall@K、NDCG@K 等指标，日志与可视化结果保存在 `./log` 和 `./visual` 目录。


## 6. 主要文件说明

- `main.py`：主训练与评测入口
- `model/MGBR.py`：MGBR模型核心实现
- `dataset.py`：数据集加载与处理
- `loss.py`：损失函数（BPR等）
- `metric.py`：评估指标
- `utils/`：辅助工具函数
- `config.py`：全局配置文件

## 7. 论文引用

如本项目对您的研究有帮助，请引用原论文：

```
@inproceedings{MGBR2024,
  title={A Multi-Interaction Graph Attention Network for Bundle Recommendation},
  author={作者列表},
  booktitle={会议/期刊名称},
  year={2025}
}
```

## 8. 贡献与反馈

欢迎提交Issue或Pull Request改进本项目。如有学术交流或合作意向，请联系作者邮箱。


