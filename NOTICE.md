# Sources and attribution

- **HoReN**: [ha11ucin8/HoReN](https://github.com/ha11ucin8/HoReN), reference commit `38bbf34d4f37a2bb1b24dbef6b8bc1fb5554fbae`. The value-memory equations, label-mean matching, Qwen tokenization conventions and evaluation semantics follow the audited implementation. This repository packages a separate HoReN/OPD implementation; it does not vendor the upstream repository or its unrelated editing methods.
- **On-policy distillation**: [On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649). Here the teacher is the pre-edit adapter and the student optimizes supervised editing plus full-vocabulary reverse KL on student-generated continuations.
- **UnKEBench**: [TrustedLLM/UnKE](https://github.com/TrustedLLM/UnKE), *Everything is Editable: Extend Knowledge Editing to Unstructured Data in Large Language Models*. The data and native Original/Para/Sub metric interpretation retain their upstream attribution.
- **MQuAKE-Remastered**: [official repository](https://github.com/henryzhongsc/MQuAKE-Remastered), code commit `349dcc50460c251ee09a6804aab444508a016485`, [dataset](https://huggingface.co/datasets/henryzhongsc/MQuAKE-Remastered) revision `b54712d4b464d7e2d4edccd4022f95ddbcb719e7`. CF6334 preparation and scoring follow its official split and answer matching rules.
- Model assets: [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) and [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
- Preservation data: [GSM8K](https://huggingface.co/datasets/openai/gsm8k) and [BIG-Bench-Hard](https://github.com/suzgunmirac/BIG-Bench-Hard).

Datasets and model weights are acquired separately and remain subject to their own upstream licenses and terms. No upstream license is inferred or replaced by this notice. This release includes no credentials, private datasets, original experiment logs or model checkpoints.
