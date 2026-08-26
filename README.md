# RSEC

Official code and data for **Residual-Scope Event Completion (RSEC)**, which
directly predicts complete circRNA–RBP binding regions. This repository contains
only the full RSEC model; baseline implementations and ablation variants are not
included.

## Contents

```text
RSEC_public/
├── README.md
├── requirements.txt
├── config.yaml
├── rsec.py
├── data.py
├── metrics.py
├── train.py
├── evaluate.py
├── predict.py
├── data/<RBP>/
├── splits_fixed_60_20_20/<RBP>/split.json
└── checkpoints/<RBP>/best_event_checkpoint.pt
```

Seven RBP-specific datasets and checkpoints are included: EIF4A3, EWSR1,
FXR1, FXR2, IGF2BP2, IGF2BP3, and MOV10. The data originate from the datasets
distributed with circGMST. In each RBP directory, `<RBP>_seq.fasta` contains
the complete circRNA sequences, while `<RBP>_train.fasta` and
`<RBP>_test.fasta` provide the source binding-region annotations. These legacy
filenames do not define the RSEC split. The authoritative sequence-level
60%/20%/20% membership is stored in `splits_fixed_60_20_20/`.

## Installation

Python 3.10 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU use, install the PyTorch build appropriate for the local CUDA version.

## Evaluate the paper checkpoints

```bash
python evaluate.py \
  --rbp EIF4A3 \
  --split test \
  --protocol seqcal \
  --output outputs/EIF4A3_test.json

python evaluate.py \
  --rbp all \
  --split test \
  --protocol seqcal \
  --output outputs/all_test.json
```

Evaluation verifies the configured checkpoint checksum and confirms that the
checkpoint was trained on the checked-in split. Coordinates in prediction
outputs are zero-based and inclusive.

## Predict new sequences

Provide any nucleotide FASTA file whose records have unique identifiers.

```bash
python predict.py \
  --rbp IGF2BP3 \
  --input your_sequences.fasta \
  --protocol seqcal \
  --output outputs/IGF2BP3_predictions.json
```

Use `--protocol native` to run the uncalibrated decoder. The default `seqcal`
protocol applies the validation-selected threshold recorded in `config.yaml`.

## Train on the fixed split

```bash
python train.py \
  --rbp EIF4A3 \
  --output outputs/EIF4A3/rsec
```

Training reads the checked-in manifest and selects the best checkpoint using
validation Event AP averaged across IoU thresholds 0.3–0.7. Test labels are not
used for model selection or SeqCal threshold selection.

## Reproducibility safeguards

`config.yaml` records the paper model parameters, validation-selected SeqCal
thresholds, split hashes, and checkpoint SHA-256 values. Training, evaluation,
and prediction reject incompatible split or checkpoint metadata rather than
silently mixing experimental settings.

Before public redistribution, the repository owner should select a code license
and confirm that redistribution of the bundled third-party datasets complies
with their source terms.
