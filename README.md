# RSEC

RSEC 用于预测全长 circRNA 中的 RBP 结合区域。本仓库提供模型代码、数据和训练好的权重。

## 安装

建议使用 Python 3.10。

```bash
pip install -r requirements.txt
```

## 评估

```bash
python evaluate.py --rbp all --split test --protocol seqcal --output outputs/all_test.json
```

程序按 `retained_test_ids.json` 选择论文使用的测试样本。将 `all` 改为对应的 RBP 名称即可单独评估。

## 预测

```bash
python predict.py --rbp IGF2BP3 --input sequences.fasta --output outputs/predictions.json
```

输入为 FASTA 文件，输出坐标从 0 开始，包含起点和终点。

## 训练

```bash
python train.py --rbp EIF4A3 --output outputs/EIF4A3/rsec
```

随附权重使用原始训练集和验证集。重新训练时，脚本会自动排除指定的异常标注记录。

## 数据

数据来自 circGMST，保存在 `data/`。训练、验证和测试的原始划分保存在 `splits_fixed_60_20_20/`。 [circGMST 原始数据](https://github.com/gyj9811/circGMST/tree/9300b5cec040b3c5eea09cd0a7acf0c5e54b2f1d/nucleotidedatasets)
