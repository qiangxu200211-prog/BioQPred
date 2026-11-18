# BioQPred

[English](README.md)

BioQPred 是用于剪接位点与 H3K27me3 预测的 DNA 序列分类模型，
结合局部序列编码、核苷酸组成特征和冻结的 DNABERT-2 表征。

本仓库提供模型实现，以及论文实验使用的数据准备、训练、评估和分析脚本。
数据集与 DNABERT-2 权重通过准备脚本下载。
仓库不包含训练后的 BioQPred 权重。

## 模型

局部编码器在卷积前加入可学习的位置嵌入，使用核大小为 3、5、7、11、15
的五个卷积分支，每个分支为 128 通道，随后通过膨胀卷积、通道注意力和空间金字塔池化
得到局部表征。组成分支接收从序列计算的 12 维特征，上下文分支对冻结的
DNABERT-2 输出进行带掩码的均值与最大池化。各分支表征拼接后用于分类。

训练脚本支持以下配置：

| 配置 | 组成 |
|---|---|
| `full` | 局部编码器 + 组成特征 + 冻结的 DNABERT-2 |
| `local_comp` | 局部编码器 + 组成特征 |
| `local_only` | 局部编码器 |
| `without_comp` | 局部编码器 + 冻结的 DNABERT-2 |
| `without_position` | 完整模型去除局部编码器的位置嵌入 |
| `without_local` | 组成特征 + 冻结的 DNABERT-2 |

`local_only` 和 `local_comp` 不需要预训练骨干模型。
教师蒸馏使用单独的训练脚本，学生模型推理时不调用教师。

## 安装

实验使用 Python 3.9.23 和 PyTorch 2.8.0。训练脚本依赖 POSIX 文件锁，
请使用 Linux 或 WSL2。训练建议使用 CUDA，功能检查可在 CPU 上运行。

```bash
git clone https://github.com/qiangxu200211-prog/BioQPred.git
cd BioQPred

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 先安装适合本机 CUDA 或 CPU 环境的 PyTorch 2.8.0。
python -m pip install -r requirements.txt
```

以下命令均在仓库根目录执行。检查源码完整性并运行 CPU 功能测试：

```bash
python scripts/check_integrity.py
python -m unittest discover -s tests -v
```

## 数据

四个任务来自
[InstaDeepAI/nucleotide_transformer_downstream_tasks_revised](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks_revised)，
固定版本为 `851f9946252e90c665cdb3cc3eedb78f1f26197c`。

| 任务 | 序列长度 | 类别数 | 训练集 | 验证集 | 测试集 |
|---|---:|---:|---:|---:|---:|
| `splice_sites_all` | 600 bp | 3 | 27,000 | 3,000 | 3,000 |
| `splice_sites_acceptor` | 600 bp | 2 | 27,000 | 3,000 | 3,000 |
| `splice_sites_donor` | 600 bp | 2 | 27,000 | 3,000 | 3,000 |
| `H3K27me3` | 1,000 bp | 2 | 27,000 | 3,000 | 3,000 |

保留官方测试集，并从官方训练集中按类别分层划出 10% 作为验证集，随机种子为 42。

```bash
python scripts/prepare_data.py
```

脚本下载源 FASTA 文件，将划分后的数据写入
`data/splits/nt_tasks/<task>/{train,dev,test}.jsonl`，并通过 SHA256
核对下载文件及重建的数据划分与实验记录是否一致。
只准备一个任务时，可添加 `--tasks splice_sites_acceptor`。
使用已有 FASTA 文件时，可添加 `--offline --upstream-dir /path/to/pinned_fasta`。

训练集与验证集可能存在基因组区间或共享基因重叠，相关检查见
[实验说明](docs/EXPERIMENTS.md)。

## 训练

### 快速运行

完成数据准备后，可先在 CPU 上运行一个小规模示例：

```bash
python revision_20261022/run_controlled_experiments.py \
  --tasks splice_sites_acceptor --variants local_comp --seeds 42 \
  --device cpu --smoke --output outputs/smoke_local
```

该命令使用 64 条训练样本和 32 条验证样本运行一轮，不进行测试集评估。

### 局部模型

下面的命令训练局部编码器与组成特征组合的模型，使用三个随机种子：

```bash
python revision_20261022/run_controlled_experiments.py \
  --tasks splice_sites_acceptor --variants local_comp --seeds 42 2024 3407 \
  --epochs 15 --patience 4 --batch-size 16 --lr 0.0001 \
  --device cuda:0 --amp --output outputs/acceptor_local_comp
```

训练使用 AdamW，初始学习率为 `1e-4`，权重衰减为 `0.01`。
验证集 MCC 停滞时降低学习率，连续四轮未改善时提前停止。
根据验证集 MCC 选择最佳模型，再对测试集进行评估。

### 完整模型

使用上下文分支的配置需要先下载 DNABERT-2 权重：

```bash
python scripts/prepare_backbone.py
python revision_20261022/run_controlled_experiments.py \
  --tasks splice_sites_acceptor --variants full --seeds 42 2024 3407 \
  --epochs 15 --patience 4 --batch-size 16 --lr 0.0001 \
  --device cuda:0 --amp --output outputs/acceptor_full
```

准备脚本逐张量校验预训练权重后，将其保存为
`models/dnabert2/model.safetensors`。也可通过
`--from-file /path/to/model.safetensors` 使用本地权重文件。
仓库中的 DNABERT-2 运行时代码采用 PyTorch attention 实现，上下文特征会缓存复用。

每次运行的结果保存在 `--output` 下按任务组织的子目录中，
包括 `best.pt`、`spec.json`、`history.json` 和 `result.json`；
进行测试集评估时还会生成 `test_predictions.csv`。
汇总三个随机种子的结果时，使用均值和样本标准差（`ddof=1`）。

## 预测

使用受控实验或教师蒸馏训练脚本生成的 `best.pt`。
输入为 JSONL，每行包含唯一的字符串 `id` 和 `sequence`。
序列使用 A/C/G/T/N 字符，长度须与对应任务的窗口一致，不需要提供标签。

```bash
python scripts/predict.py \
  --checkpoint /path/to/best.pt \
  --input /path/to/sequences.jsonl \
  --output outputs/predictions.csv --device cpu
```

输出 CSV 包含 `id`、`prediction` 和各类别的概率。
带上下文分支的模型在处理新序列时仍需要 DNABERT-2 权重。
基线比较脚本产生的 checkpoint 使用不同格式，由 `run_fair_baselines.py` 完成测试集评估。

## 其他实验

[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) 提供组件消融、教师蒸馏、
基线比较、卷积核与通道数分析、基因组检查、序列遮蔽和效率测量的命令与前置条件。
CAMP、TFBS 和 NeuGSE 对照模型根据本项目任务进行了适配；
OpenSpliceAI 对照采用候选窗口分类适配方式。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `src/models/` | BioQPred 与基线模型定义 |
| `revision_20261022/` | 训练、评估与分析脚本 |
| `scripts/` | 数据及权重准备、预测与完整性检查 |
| `models/dnabert2/` | DNABERT-2 配置、分词器与运行时代码 |
| `configs/` | 数据版本、划分校验值与模型元数据 |
| `tests/` | CPU 功能测试 |
| `docs/EXPERIMENTS.md` | 详细实验设置 |
| `third_party/` | 第三方来源与许可说明 |

## 许可与致谢

仓库暂未指定整体许可协议。第三方代码与依赖保留各自的许可：
DNABERT-2 运行时代码采用 Apache-2.0，外部可选依赖 OpenSpliceAI 采用 GPL-3.0。
具体来源和许可见 [third_party/NOTICE.md](third_party/NOTICE.md)。
