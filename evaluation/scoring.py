"""Scoring helpers for ASR and SLU recipes.

Expected predictions JSONL format:

    {"id": "...", "ref": "...", "hyp": "..."}

Authors
-------
 * Haroun Elleuch, 2026
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import jiwer


Prediction = dict[str, str]

WHISPER_TAG_PATTERN = re.compile(r"<\|.*?\|>")
SLU_SPAN_PATTERN = re.compile(r"<(?P<label>[^<>/\s]+)>\s*(?P<value>.*?)\s*>")


@dataclass(frozen=True)
class SLUSpan:
    """One SLU span."""

    label: str
    value: str


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace."""
    return " ".join(text.strip().split())


def clean_whisper_tokens(text: str) -> str:
    """Remove Whisper special tokens while keeping SLU tags."""
    return normalize_whitespace(WHISPER_TAG_PATTERN.sub("", text))


def extract_slu_spans(text: str) -> list[SLUSpan]:
    """Extract SLU spans from the recipe annotation format.

    Example
    -------
    '<time> السبعة و النصف متاع الصباح >'
    becomes:
    SLUSpan(label='time', value='السبعة و النصف متاع الصباح')
    """
    text = clean_whisper_tokens(text)

    spans: list[SLUSpan] = []

    for match in SLU_SPAN_PATTERN.finditer(text):
        label = normalize_whitespace(match.group("label"))
        value = normalize_whitespace(match.group("value"))
        spans.append(SLUSpan(label=label, value=value))

    return spans


def remove_slu_tags(text: str) -> str:
    """Remove SLU tags while keeping the spoken words.

    Example
    -------
    'بلاهي فيقني <time> السبعة و النصف متاع الصباح >'
    becomes:
    'بلاهي فيقني السبعة و النصف متاع الصباح'
    """
    text = clean_whisper_tokens(text)

    def replace_span(match: re.Match) -> str:
        return f" {match.group('value')} "

    return normalize_whitespace(SLU_SPAN_PATTERN.sub(replace_span, text))


def make_concept_value_token(span: SLUSpan) -> str:
    """Build one whitespace-free concept-value token for CVER.

    Example
    -------
    SLUSpan(label='time', value='السبعة و النصف')
    becomes:
    'time=السبعة_و_النصف'
    """
    label = normalize_whitespace(span.label)
    value = normalize_whitespace(span.value).replace(" ", "_")
    return f"{label}={value}"


def load_predictions_jsonl(predictions_file: str | Path) -> list[Prediction]:
    """Load predictions from a JSONL file."""
    predictions_file = Path(predictions_file)

    with open(predictions_file, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_predictions_jsonl(
    predictions_file: str | Path,
    predictions: list[Prediction],
) -> None:
    """Write predictions to JSONL without changing the record structure."""
    predictions_file = Path(predictions_file)
    predictions_file.parent.mkdir(parents=True, exist_ok=True)

    with open(predictions_file, "w", encoding="utf-8") as f:
        for prediction in predictions:
            f.write(json.dumps(prediction, ensure_ascii=False) + "\n")


def compute_wer_cer(
    refs: list[str],
    hyps: list[str],
) -> dict[str, float]:
    """Compute corpus-level WER and CER using jiwer."""
    refs = [normalize_whitespace(ref) for ref in refs]
    hyps = [normalize_whitespace(hyp) for hyp in hyps]

    return {
        "WER": 100.0 * jiwer.wer(refs, hyps),
        "CER": 100.0 * jiwer.cer(refs, hyps),
    }


def compute_asr_metrics(
    predictions: list[Prediction],
) -> dict[str, float]:
    """Compute corpus-level WER and CER for an ASR task.

    This assumes references and hypotheses are plain transcriptions.
    """
    refs = [clean_whisper_tokens(item["ref"]) for item in predictions]
    hyps = [clean_whisper_tokens(item["hyp"]) for item in predictions]

    return compute_wer_cer(refs, hyps)


def compute_slu_asr_metrics(
    predictions: list[Prediction],
) -> dict[str, float]:
    """Compute corpus-level WER and CER for an SLU task.

    SLU tags are removed before computing WER/CER, while the spoken words
    inside the tagged spans are kept.
    """
    refs = [remove_slu_tags(item["ref"]) for item in predictions]
    hyps = [remove_slu_tags(item["hyp"]) for item in predictions]

    return compute_wer_cer(refs, hyps)


def compute_coer(
    predictions: list[Prediction],
) -> float:
    """Compute corpus-level Concept Error Rate.

    CoER is WER over concept-label sequences.

    Example
    -------
    '<time> السبعة >' becomes:
    'time'
    """
    refs: list[str] = []
    hyps: list[str] = []

    for item in predictions:
        ref_spans = extract_slu_spans(item["ref"])
        hyp_spans = extract_slu_spans(item["hyp"])

        refs.append(" ".join(span.label for span in ref_spans))
        hyps.append(" ".join(span.label for span in hyp_spans))

    if not any(refs):
        return 0.0

    return 100.0 * jiwer.wer(refs, hyps)


def compute_cver(
    predictions: list[Prediction],
) -> float:
    """Compute corpus-level Concept-Value Error Rate.

    CVER is WER over concept-value sequences. Each annotated span is treated
    as one unit.

    Example
    -------
    '<time> السبعة و النصف >' becomes:
    'time=السبعة_و_النصف'
    """
    refs: list[str] = []
    hyps: list[str] = []

    for item in predictions:
        ref_spans = extract_slu_spans(item["ref"])
        hyp_spans = extract_slu_spans(item["hyp"])

        refs.append(
            " ".join(make_concept_value_token(span) for span in ref_spans)
        )
        hyps.append(
            " ".join(make_concept_value_token(span) for span in hyp_spans)
        )

    if not any(refs):
        return 0.0

    return 100.0 * jiwer.wer(refs, hyps)


def compute_slu_metrics(
    predictions: list[Prediction],
) -> dict[str, float]:
    """Compute corpus-level WER, CER, CoER, and CVER for an SLU task.

    WER and CER are computed after removing SLU tags.
    CoER and CVER are computed from SLU spans.
    """
    metrics = compute_slu_asr_metrics(predictions)

    metrics.update(
        {
            "CoER": compute_coer(predictions),
            "CVER": compute_cver(predictions),
        }
    )

    return metrics


def compute_metrics_from_jsonl(
    predictions_file: str | Path,
    task: str,
) -> dict[str, float]:
    """Compute metrics from a predictions JSONL file.

    Arguments
    ---------
    predictions_file
        Path to predictions JSONL file.
    task
        Either "asr" or "slu".
    """
    predictions = load_predictions_jsonl(predictions_file)

    if task == "asr":
        return compute_asr_metrics(predictions)

    if task == "slu":
        return compute_slu_metrics(predictions)

    raise ValueError(f"Unsupported task: {task}. Expected 'asr' or 'slu'.")


def write_metrics_txt(
    metrics_file: str | Path,
    metrics: dict[str, float],
) -> None:
    """Write metrics to a text file."""
    metrics_file = Path(metrics_file)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_file, "w", encoding="utf-8") as f:
        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name}: {metric_value:.2f}\n")