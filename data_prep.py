"""Data preparation for the SLURP-TN SpeechBrain recipe."""

import argparse
import csv
import re
from io import BytesIO
from pathlib import Path

import soundfile as sf
from datasets import Audio, Dataset, DatasetDict, load_dataset
from speechbrain.utils.logger import get_logger
from tqdm import tqdm


logger = get_logger(__name__)

HF_REPO_ID = "Elyadata/SLURP-TN"
HF_TUN_SLU_COLUMN = "tun_slu_annoation"
CSV_TUN_SLU_COLUMN = "tun_slu_annotation"

SLU_BRACKET_PATTERN = re.compile(
    r"\[(?P<label>[^\[\]:]+)\s*:\s*(?P<value>[^\[\]]+?)\]"
)


def prepare_slurptn(data_folder: str | Path) -> None:
    """Download SLURP-TN and prepare SpeechBrain CSV manifests.

    Arguments
    ---------
    data_folder
        Root data directory. Hugging Face files, exported wav files,
        and CSV manifests are stored under this directory.
    """
    data_folder = Path(data_folder)
    wav_folder = data_folder / "wav"
    manifest_folder = data_folder / "manifests"

    data_folder.mkdir(parents=True, exist_ok=True)
    wav_folder.mkdir(parents=True, exist_ok=True)
    manifest_folder.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset from Hugging Face: %s", HF_REPO_ID)

    dataset = load_dataset(
        HF_REPO_ID,
        cache_dir=str(data_folder),
    )

    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected DatasetDict, got {type(dataset)}.")

    for split in dataset.keys():
        csv_path = manifest_folder / f"{split}.csv"

        if csv_path.exists():
            logger.info("Skipping %s because %s already exists.", split, csv_path)
            continue

        prepare_split(
            dataset=dataset[split],
            split=split,
            audio_folder=wav_folder / split,
            csv_path=csv_path,
        )


def prepare_split(
    dataset: Dataset,
    split: str,
    audio_folder: str | Path,
    csv_path: str | Path,
) -> None:
    """Prepare one SLURP-TN split.

    Arguments
    ---------
    dataset
        Hugging Face dataset split.
    split
        Split name.
    audio_folder
        Directory where split audio files will be saved.
    csv_path
        Output CSV manifest path.
    """
    audio_folder = Path(audio_folder)
    csv_path = Path(csv_path)

    audio_folder.mkdir(parents=True, exist_ok=True)
    validate_columns(dataset, split)

    dataset = dataset.cast_column("audio", Audio(decode=False))

    rows: list[dict[str, str]] = []

    logger.info("Preparing split '%s' with %d examples.", split, len(dataset))

    for idx, example in enumerate(tqdm(dataset, desc=f"Preparing {split}")):
        utt_id = f"slurptn_{split}_{idx:06d}"

        audio = example["audio"]
        wav_path = audio_folder / f"{utt_id}.wav"

        duration = save_audio(audio=audio, wav_path=wav_path)

        rows.append(
            {
                "ID": utt_id,
                "duration": f"{duration:.3f}",
                "wav": str(wav_path.resolve()),
                "tun_transcription": normalize_whitespace(
                    example["tun_transcription"]
                ),
                CSV_TUN_SLU_COLUMN: convert_slu_annotation(
                    example[HF_TUN_SLU_COLUMN]
                ),
            }
        )

    write_csv_manifest(csv_path, rows)
    logger.info("Saved %s", csv_path)


def save_audio(audio: dict, wav_path: str | Path) -> float:
    """Save one Hugging Face audio example and return its duration.

    Arguments
    ---------
    audio
        Hugging Face audio dictionary with decode=False.
    wav_path
        Path where the audio file will be saved.

    Returns
    -------
    float
        Audio duration in seconds.
    """
    wav_path = Path(wav_path)

    audio_bytes = audio.get("bytes")
    audio_path = audio.get("path")

    if audio_bytes is not None:
        if not wav_path.exists():
            with open(wav_path, "wb") as audio_file:
                audio_file.write(audio_bytes)

        info = sf.info(BytesIO(audio_bytes))
        return info.frames / info.samplerate

    if audio_path is not None:
        source_path = Path(audio_path)

        if not wav_path.exists():
            data, sample_rate = sf.read(source_path)
            sf.write(wav_path, data, sample_rate)

        info = sf.info(wav_path)
        return info.frames / info.samplerate

    raise ValueError("Audio example does not contain either 'bytes' or 'path'.")


def convert_slu_annotation(annotation: str) -> str:
    """Convert SLURP-TN bracket annotations to recipe SLU tags.

    Example
    -------
    'بلاهي فيقني [time : السبعة و النصف متاع الصباح]'
    becomes:
    'بلاهي فيقني <time> السبعة و النصف متاع الصباح >'
    """

    def replace_span(match: re.Match) -> str:
        label = normalize_whitespace(match.group("label"))
        value = normalize_whitespace(match.group("value"))
        return f"<{label}> {value} >"

    return normalize_whitespace(SLU_BRACKET_PATTERN.sub(replace_span, annotation))


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in a text string."""
    return " ".join(text.strip().split())


def validate_columns(dataset: Dataset, split: str) -> None:
    """Check that the expected SLURP-TN columns exist."""
    required_columns = [
        "audio",
        "tun_transcription",
        HF_TUN_SLU_COLUMN,
    ]

    missing_columns = [
        column for column in required_columns if column not in dataset.column_names
    ]

    if missing_columns:
        raise ValueError(
            f"Split '{split}' is missing columns: {missing_columns}. "
            f"Available columns are: {dataset.column_names}"
        )


def write_csv_manifest(
    csv_path: str | Path,
    rows: list[dict[str, str]],
) -> None:
    """Write a SpeechBrain CSV manifest."""
    fieldnames = [
        "ID",
        "duration",
        "wav",
        "tun_transcription",
        CSV_TUN_SLU_COLUMN,
    ]

    with open(csv_path, mode="w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare SLURP-TN SpeechBrain manifests."
    )

    parser.add_argument(
        "--data_folder",
        type=str,
        default="data",
        help="Root directory where dataset files, wavs, and manifests are saved.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    prepare_slurptn(data_folder=args.data_folder)