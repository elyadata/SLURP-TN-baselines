# SLURP-TN Baselines

SpeechBrain baseline recipes for **SLURP-TN**, a Tunisian Arabic spoken language understanding dataset.

This repository provides baseline systems for:

- **ASR**: Automatic Speech Recognition using Tunisian Arabic transcriptions.
- **SLU / Slot Filling**: Speech-to-semantics prediction using inline slot annotations.

The recipes are based on SpeechBrain and Hugging Face Whisper models.

## Repository structure

```text
.
├── ASR
│   └── whisper
│       ├── hparams
│       │   └── whisper.yaml
│       └── train_whisper.py
├── Evaluation
│   ├── README.MD
│   └── scoring.py
├── SLU
│   └── slot_filling
│       ├── hparams
│       │   └── whisper.yaml
│       ├── setup_sf_tokenizer.py
│       └── train_whisper.py
├── data_prep.py
├── requirements.txt
└── README.MD
```

## Installation

Create and activate a Python environment:

```bash
conda create -n slurp python=3.12
conda activate slurp
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Data preparation

The data preparation script downloads the SLURP-TN dataset from Hugging Face and creates SpeechBrain-compatible CSV manifests.

Run from the repository root:

```bash
python data_prep.py
```

By default, this creates:

```text
data/
├── wav/
│   ├── train/
│   ├── validation/
│   └── test/
└── manifests/
    ├── train.csv
    ├── validation.csv
    └── test.csv
```

The script also stores Hugging Face dataset cache files under the same `data/` directory.

You can explicitly specify the data folder:

```bash
python data_prep.py --data_folder data
```

## Manifest format

The generated CSV manifests contain the following columns:

```text
ID,duration,wav,tun_transcription,tun_slu_annotation
```

Column descriptions:

| Column | Description |
|---|---|
| `ID` | Unique utterance identifier |
| `duration` | Audio duration in seconds |
| `wav` | Path to the exported WAV file |
| `tun_transcription` | Tunisian Arabic transcription |
| `tun_slu_annotation` | Tunisian Arabic slot-filling annotation |

The original Hugging Face SLU annotation format is converted during data preparation.

Original format:

```text
بلاهي فيقني [time : السبعة و النصف متاع الصباح]
```

Recipe format:

```text
بلاهي فيقني <time> السبعة و النصف متاع الصباح >
```

## Running data preparation from the training script

Both ASR and SLU recipes can run data preparation automatically.

In the YAML config:

```yaml
data_folder: ../../data
skip_prep: False
```

When `skip_prep: False`, the training script calls:

```python
prepare_slurptn(data_folder=hparams["data_folder"])
```

After the manifests and WAV files have been created once, you can set:

```yaml
skip_prep: True
```

to avoid repeating data preparation.

## ASR baseline

The ASR recipe trains Whisper on the `tun_transcription` column.

Example YAML settings:

```yaml
target_column: tun_transcription
task: asr
```

Run from the repository root:

```bash
python ASR/whisper/train_whisper.py ASR/whisper/hparams/whisper.yaml
```

The ASR recipe produces:

```text
results/<experiment_id>/dev_scores/predictions.jsonl
results/<experiment_id>/dev_scores/metrics.txt
results/<experiment_id>/test_scores/predictions.jsonl
results/<experiment_id>/test_scores/metrics.txt
```

ASR metrics:

- WER
- CER

Metrics are computed using JiWER through `Evaluation/scoring.py`.

## SLU / Slot-Filling baseline

The SLU recipe trains Whisper on the `tun_slu_annotation` column.

Example YAML settings:

```yaml
target_column: tun_slu_annotation
task: slu
```

Run from the repository root:

```bash
python SLU/slot_filling/train_whisper.py SLU/slot_filling/hparams/whisper.yaml
```

The SLU recipe uses `setup_sf_tokenizer.py` to extend the Whisper tokenizer with slot-filling tokens inferred from the CSV manifests.

For example, if the dataset contains:

```text
<time> السبعة و النصف متاع الصباح >
```

then the tokenizer receives special tokens such as:

```text
<time>
>
```

The SLU recipe produces:

```text
results/<experiment_id>/dev_scores/predictions.jsonl
results/<experiment_id>/dev_scores/metrics.txt
results/<experiment_id>/test_scores/predictions.jsonl
results/<experiment_id>/test_scores/metrics.txt
```

SLU metrics:

- WER
- CER
- CoER
- CVER

WER and CER are computed after removing the slot tags while keeping the spoken words.

## Prediction file format

Both ASR and SLU recipes export predictions in JSONL format:

```json
{"id": "slurptn_test_000000", "ref": "...", "hyp": "..."}
```

This format is intentionally simple and is not modified by the scoring script.

It can also be reused for external evaluation tools such as SCTK/SCLITE after converting the references and hypotheses to the required format.

## Evaluation

The shared evaluation helper is located at:

```text
evaluation/scoring.py
```

It uses JiWER to compute corpus-level scores from `predictions.jsonl`.

For details about metric conventions, see:

```text
evaluation/README.MD
```

## Notes

- The ASR recipe uses the original Whisper tokenizer.
- The SLU recipe creates or loads a custom tokenizer with slot-filling tokens.
- Data preparation uses `Audio(decode=False)` to avoid TorchCodec/FFmpeg issues.
- The generated manifests should be kept stable once experiments start.

## Citation

If you use this repository, the SLURP-TN dataset, or the baseline recipes in your work, please cite:

```bibtex
@misc{elleuch2026slurptn,
  title         = {SLURP-TN: Resource for Tunisian Dialect Spoken Language Understanding},
  author        = {Elleuch, Haroun and Mdhaffar, Salima and Est{\`e}ve, Yannick and Bougares, Fethi},
  year          = {2026},
  eprint        = {2603.21940},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2603.21940}
}
```