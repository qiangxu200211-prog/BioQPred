# BioQPred 论文源码

本包用于发布 BioQPred 模型及论文修订实验的复现代码。
完整技术说明和全部实验命令见 [README.md](README.md) 与
[实验说明](docs/EXPERIMENTS.md)。

## 包内内容

- BioQPred 的局部、局部加组成、完整上下文配置和模块消融实现。
- 原始训练、验证集选模、公平基线、教师蒸馏和专业剪接对照入口。
- 数据版本固定、原始划分重建、源码校验、预测与效率测量辅助入口。
- Python 依赖、CPU 功能测试及第三方代码许可说明。

不包含数据集、训练权重、大型特征缓存、服务器密码、内部调度记录、稿件、审稿意见，
也不包含无关的 ReCaRe 任务。原实验目录和论文内容没有因本次打包而改变。

## 安装与快速检查

训练入口使用 Linux 文件锁，请使用 Linux 或 WSL2，并始终在仓库根目录运行。
原科学环境为 Python 3.9.23、PyTorch 2.8.0；按自己的 CUDA 环境选择匹配的 PyTorch 安装包。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_integrity.py
python -m unittest discover -s tests -v
```

## 数据准备

```bash
python scripts/prepare_data.py
```

程序从固定版本的公开 Nucleotide Transformer 下游数据源下载八个 FASTA，
校验原文件并重建四个任务的训练、验证、测试集。每任务分别为 27,000、3,000、3,000 条，
共 132,000 条任务记录；不能据此声称跨任务序列全部唯一。
剪接任务窗口为 600 bp，H3K27me3 为 1,000 bp。
官方测试集保持原样，验证集由官方训练集按类别分层、随机状态 42 切出 10%。
十二个结果文件须与原实验的 SHA256 完全一致，否则停止，不覆盖冲突文件。

## 先跑一个功能测试

```bash
python revision_20261022/run_controlled_experiments.py \
  --tasks splice_sites_acceptor --variants local_comp --seeds 42 \
  --device cpu --smoke --output outputs/smoke_local
```

这里仅使用 64 条训练数据、32 条验证数据，运行一轮，不评估测试样本。
它用于确认程序可运行，不能作为论文实验结果。
`local_comp` 和 `local_only` 不依赖预训练权重。

## 正式训练

```bash
python revision_20261022/run_controlled_experiments.py \
  --tasks splice_sites_acceptor --variants local_comp --seeds 42 2024 3407 \
  --device cuda:0 --amp --output outputs/acceptor_local_comp
```

默认学习率 1e-4、batch size 16、最多 15 轮、早停 patience 4，按验证集 MCC 保存模型。
卷积默认五个分支，核大小为 3/5/7/11/15，**每分支**128 通道。
完整上下文配置先运行 `python scripts/prepare_backbone.py` 获取外部 DNABERT-2 权重，
再将训练参数改为 `--variants full`。程序保留原实验的上下文实现，并逐张量验证权重。
权重本身不在源码压缩包内。

教师消融、公平调参、OpenSpliceAI、NeuGSE、参数量控制和效率测量需要各自的前置条件，
不能仅改一个参数就混为同一种实验。具体命令见英文实验说明。
OpenSpliceAI 依赖需要单独按固定 commit 下载；它在本项目中是候选窗口分类适配版，
不是直接使用原论文的全基因组预训练集成模型。

## 使用已训练模型预测

```bash
python scripts/predict.py \
  --checkpoint /path/to/trusted/best.pt \
  --input /path/to/sequences.jsonl --output outputs/predictions.csv --device cpu
```

该入口接收本包受控消融或教师学生训练产生的 checkpoint。
输入每行包含唯一 `id` 与 `sequence`，不需要标签；序列长度须符合任务窗口。
输出预测类别和各类别概率。请只加载可信 checkpoint。

## 实验统计与发布注意事项

三种子应报告均值和样本标准差，不能只报告最好种子，也不能凭近似比例修改标准差。
三个种子使用相同 3,000 条测试样本，9,000 行预测不是 9,000 个独立样本。
保留原始失败与较差结果；本包不会自动重试失败或默默降低科学核验门槛。
原始训练/验证数据存在区间或共享基因重叠，不能写成所有划分完全生物学独立。
缓存输入吞吐量不等于包含新特征提取的端到端吞吐量，参数少也不自动等于速度快。

目前**尚未替作者选择仓库整体开源协议**。DNABERT-2 运行时代码保留原 Apache-2.0 许可；
OpenSpliceAI 为独立 GPL-3.0 依赖，不能被仓库新协议覆盖。作者确认许可、署名和论文引用后再补齐。
本次仅在本地整理源码，没有自动创建或推送 GitHub 仓库。
