# 指标计算口径

## ZsRE

`evaluation._edit_prediction` 保留原 HoReN 自回归 token 指标：生成预算为带前导空格目标的 token 数；提取续写、解码、去左侧空白、重新编码并截取目标 token 数。预测长度不足时为 0，否则比较每个位置；各案例先求均值，再对案例求宏平均。

Rel 使用编辑原问题，Gen 使用改写问题。它们不是整句 exact match，也不是 teacher-forcing accuracy。空目标按上游实现计 1；数据准备应保留真实有效目标。

Loc 比较无关问题的编辑前后生成 token。保留上游从完整生成序列尾部取目标长度的约定；提前 EOS 时可能包含提示尾部。匹配位置数除以两序列最大长度，均为空时为 1。必须使用未编辑模型的参考输出，不能以 ground truth 准确率替代。

## UnKEBench

Original 对应 Rel，Para 对应 Gen；每项均报告以下四类分数，而不是混合成一个数。

- BLEU：大小写敏感的空白分词 unigram clipped precision，乘短句惩罚 `exp(1 - len(reference)/len(prediction))`；预测更长时惩罚为 1。空文本重合度为 0。
- ROUGE-1/2/L：使用 `rouge==1.0.1` 的 recall 字段 `r`。空预测或仅句点预测的重合度为 0。
- MiniLM cosine：用固定 `all-MiniLM-L6-v2` 分别编码参考答案和预测，L2 归一化后计算对应行点积。保留负数；只修正浮点舍入越界。原生标签为 `Bert Score`。
- Sub：对子问题分别计算 ROUGE recall，先在案例内平均，再跨有子问题的案例平均。缺失子问题不计 0。

当前上游 HoReN/UnKEBench 评测没有原生 locality；输出 `Loc=null`。数据中的 MMLU 问题仅用于保持池排除，不被私自改造成 Loc。

编辑和生成保留 Qwen 原生聊天模板。监督 token 化保留原实现的前导空格和 EOS mask 行为；生成贪心、batch=1、`use_cache=False`，默认 512 新 token。

## MQuAKE-Remastered CF6334

使用固定官方划分。`train_edited` 优先；`test_edited` 排除与训练案例的重叠；`test_unedited` 组成未编辑组。训练仅使用 train_edited 的唯一编辑事实。Test Edited 案例不额外提供训练事实；若其所需编辑事实不在训练集合中，准备阶段报错。

已编辑组以新答案判分，未编辑组以原答案判分。答案与别名统一大写后整条相等才命中，不使用 substring、外部裁判或按金标准修正预测。每案例最多问三种表述，首个命中后停止；失败必须尝试全部三种。准确率为正确案例数除以案例数。

汇总 `Total / Train Edited / Test Edited / Unedited` 与 2/3/4-hop 分组。Total 是所有案例的微平均，不能代替 edited accuracy。官方两档的样本集合和组分布不同，不能只由 Total 下降判断更多编辑导致退化。

CF6334 固定版本中，1000 档有 6 个训练案例的标注链与编辑事实冲突；完整官方分数保留它们，并额外报告 `path_consistent_subset`。这个子集由准备阶段决定，不依赖模型结果。它不是原文表格的分母。

零样本直接回答、首行提取、64-token 上限属于本项目推理设置。分组和评分规则对齐 [官方实现](https://github.com/henryzhongsc/MQuAKE-Remastered/blob/349dcc50460c251ee09a6804aab444508a016485/eval/mquake_remastered/mquake_dataset.py)，不宣称复现原论文方法的分数。
