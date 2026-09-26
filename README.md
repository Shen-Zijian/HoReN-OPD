# HoReN + OPD

HoReN 和 HoReN＋逐轮 On-Policy Distillation 的独立运行版本，包含 ZsRE、UnKEBench 与 MQuAKE-Remastered CF6334 的数据准备、训练、评测和预测重算。

默认模型为 Qwen2.5-7B-Instruct，seed=42，batch size=1。模型权重在本地加载，训练期间不下载文件。代码不依赖另一个 HoReN 源码目录，不包含数据集、模型权重、实验结果、登录信息或服务器脚本。

## 安装

Python 3.10+；参考训练环境为 Linux、A100 80GB、PyTorch 2.7.0、Transformers 4.57.1。使用独立虚拟环境，先安装适合本机 CUDA 的 PyTorch 2.7.x，再安装本项目。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[data,evaluation,test]'
python -m pytest -q
```

CPU 测试使用随机初始化的小型 Qwen，不下载大模型。正式 Qwen7B 实验需要足够显存；不要同时启动多个实验占用同一张 GPU。

## 文件结构

| 文件 | 功能 |
|---|---|
| `horen_opd/core.py` | Hopfield 检索、codebook 槽位与残差注入 |
| `horen_opd/training.py` | HoReN 监督编辑、冻结教师、学生采样和反向 KL |
| `horen_opd/config.py` | 可移植配置与评测检查点 |
| `horen_opd/data.py` | ZsRE、UnKEBench 加载，历史重放与保持池隔离 |
| `horen_opd/evaluation.py` | ZsRE Rel/Gen/Loc、UnKE 原生指标 |
| `horen_opd/mquake.py` | CF6334 划分、事实去重、答案/别名精确匹配 |
| `horen_opd/mquake_evaluation.py` | 多跳生成、逐题预测与分组汇总 |
| `horen_opd/run.py` | 单次独立训练、评测、完整状态恢复 |
| `horen_opd/score.py` | 从已保存预测重新计算指标 |
| `horen_opd/download.py` | 显式下载固定版本模型和 CF6334 |
| `horen_opd/prepare_data.py` | 准备固定 GSM8K/BBH 保持池 |
| `horen_opd/prepare_mquake.py` | 生成 CF6334 训练与评测计划 |

## 方法

- `baseline`：原 HoReN value 模式，默认每条编辑最多 50 步 Adam，学习率 1.0。
- `preserve`：先完成相同的 HoReN 编辑，再执行一步 OPD；目标为当前编辑 CE 加 `KL(student || teacher)`，学习率 1e-3、KL 权重 1。

教师保存编辑前的适配器状态。学生从自己的完整词表分布采样最多 256 token，教师在同一学生前缀上打分。基础模型和检索键冻结，只更新 codebook 值。每轮最多两条保持提示：有历史时取一条不冲突的历史编辑，再补保持池问题；无历史时取两条保持池问题。评测改写和测试答案不用于重放。

所有前向使用 `use_cache=False`，保留原实现的查询位置和 token 化行为。没有周期压缩、LoRA、95% 验收门槛或通用推理评测。保持池用于 OPD 训练，不代表测量了推理能力保持。

## 准备模型与数据

所有命令从仓库根目录执行；配置中的相对路径也相对当前工作目录。可以修改配置以指向已有本地文件。

```bash
python -m horen_opd.download qwen --output-dir models/qwen2.5-7b-instruct
python -m horen_opd.download minilm --output-dir models/all-MiniLM-L6-v2
python -m horen_opd.download mquake --output-dir data/mquake_raw
```

下载入口固定以下修订，并保存文件哈希：

| 资源 | 固定版本 |
|---|---|
| Qwen/Qwen2.5-7B-Instruct | `a09a35458c702b33eeacc393d103063234e8bc28` |
| sentence-transformers/all-MiniLM-L6-v2 | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| henryzhongsc/MQuAKE-Remastered | `b54712d4b464d7e2d4edccd4022f95ddbcb719e7` |

ZsRE 和 UnKEBench 按上游格式自行准备：

```text
data/
  ZsRE/
    zsre_edit_data.json
    zsre_train_data.json
  UnKE/
    final_data_v3.json
```

ZsRE 文件名与字段遵循 [HoReN 数据说明](https://github.com/ha11ucin8/HoReN#3-download-datasets)；目标使用 `answers[0]`，保留原始顺序。UnKEBench 数据来自 [TrustedLLM/UnKE](https://github.com/TrustedLLM/UnKE)，使用 `data/final_data_v3.json`；数据下载和使用须遵循其上游条款。

OPD 使用独立准备的保持池。`--n 1000` 隔离前 1000 条编辑及评测提示，使 N500/N1000 可以共用同一池。基线不需要保持池。

```bash
python -m horen_opd.prepare_data download --output-dir data/reasoning_raw
python -m horen_opd.prepare_data build --source-json data/reasoning_raw/sources.json --dataset zsre --data-dir data/ZsRE --n 1000 --output data/reasoning_zsre.json
python -m horen_opd.prepare_data build --source-json data/reasoning_raw/sources.json --dataset unke --data-dir data/UnKE --n 1000 --output data/reasoning_unke.json
python -m horen_opd.prepare_mquake --input data/mquake_raw/CF6334.parquet --reasoning data/reasoning_unke.json --settings 100 1000 --output-dir data/mquake
```

GSM8K 固定修订为 `740312add88f781978c0658806c59bc2815b9866`；BBH 为 `9ee07bd481feebf959a6b59d61ea57bdcf30964d`。只使用准备文件的 `train`；保留的 `dev/test` 不参与训练或本项目评测。

## 运行

先检查配置和输入哈希，不加载模型：

```bash
python -m horen_opd.run --config configs/zsre.json --method preserve --n 500 --check
```

每个命令都会独立加载原模型。顺序执行同一数据集的 HoReN 与 OPD：

```bash
python -m horen_opd.run --config configs/zsre.json --method baseline --n 500
python -m horen_opd.run --config configs/zsre.json --method preserve --n 500
python -m horen_opd.run --config configs/unke.json --method baseline --n 500
python -m horen_opd.run --config configs/unke.json --method preserve --n 500
python -m horen_opd.run --config configs/mquake_100.json --method baseline
python -m horen_opd.run --config configs/mquake_100.json --method preserve
```

ZsRE/UnKE 的 N1000 使用 `--n 1000`。MQuAKE 的 1000 档使用 `configs/mquake_1000.json`，不要只修改 `--n` 而继续读取 100 档计划。

运行输出为 `runs/<dataset>_<method>_n<N>_s<seed>/`；MQuAKE 使用 `e<setting>`。已有输出默认拒绝覆盖。失败后修复运行环境，再用同一配置和 `--resume` 恢复：

```bash
python -m horen_opd.run --config configs/zsre.json --method preserve --n 500 --resume
```

恢复要求配置、代码和输入内容一致。检查点包含适配器、抽样状态和随机数状态；它不包含冻结基础模型。只加载自己生成且可信的检查点。ZsRE/UnKE 默认每条编辑保存，MQuAKE 默认每 50 条保存；后者故障时会从最近已保存边界重新执行少量编辑。旧实验版本的检查点格式不兼容本精简版。

## 指标

数值通常为 0–1，显示百分比时乘以 100；MiniLM cosine 的定义域为 −1–1。

| 数据集 | Rel | Gen | Loc | 其他 |
|---|---|---|---|---|
| ZsRE | 原问题自回归 token accuracy | 改写问题同类准确率 | 无关问题编辑前后输出 token 一致率 | 无 portability |
| UnKEBench | Original 的 BLEU、ROUGE-1/2/L、MiniLM cosine | Para 的同类指标 | `null` / N/A，原生协议未定义 | Sub 的 ROUGE-1/2/L |
| MQuAKE CF6334 | 不套用 Rel/Gen/Loc | — | — | Total、Train Edited、Test Edited、Unedited 多跳准确率 |

UnKE 的 ROUGE 使用 recall，BLEU 使用上游 BLEU-1 风格实现。上游字段 `Bert Score` 实际是 MiniLM 句向量余弦相似度，不是标准 BERTScore。Sub 是子问题指标，不能当作 Loc。详情见 [指标定义](docs/metrics.md)。

ZsRE 每 100 条评估截至该点的全部编辑，并在训练前保存原模型 locality 参考。UnKE N500 检查点为 1/10/30/100/120/500，N1000 再加 1000，无编辑前全量初评。UnKE 默认贪心生成最多 512 token。

MQuAKE 只从官方 `train_edited` 获取编辑事实，按事实去重；setting 100/1000 是划分名称，不等于实际训练事实条数。每个案例三种问法任一种与答案或别名忽略大小写精确匹配即成功。使用 Qwen chat 零样本直接回答，最多 64 token，取首行。完整官方分组与已知推理链冲突排除子集分开报告，附加单跳 token accuracy。该协议与原文的评分规则对齐，但不是原文模型、提示和推理流程的复现。

## 从预测重算

不重新运行已编辑大模型；UnKE 的语义相似度仍需本地 MiniLM。

```bash
python -m horen_opd.score --dataset zsre --predictions runs/zsre_preserve_n500_s42/evaluations/0500.json
python -m horen_opd.score --dataset unke --predictions runs/unke_preserve_n500_s42/evaluations/0500.json --similarity-model models/all-MiniLM-L6-v2
python -m horen_opd.score --dataset mquake --predictions runs/mquake_preserve_e100_s42/multihop_predictions.jsonl --cases runs/mquake_preserve_e100_s42/multihop_cases.json
```

主要产物：`final.json`、`evaluations/*.json`、逐题预测、`edit_logs.json`、`checkpoints/latest.json`、`manifest.json`。Manifest 记录配置、输入、源码与产物 SHA256。

## 来源和验证

实现来源、协议修订和归属见 [来源说明](NOTICE.md)。验证范围见 [验证记录](docs/validation.md)。本仓库提供可运行代码，不附带新的性能提升主张。
