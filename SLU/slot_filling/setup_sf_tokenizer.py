"""Setup a Whisper tokenizer for SLURP-TN slot filling.

Authors
-------
 * Haroun Elleuch, 2026
"""

import csv
import re
from pathlib import Path

from speechbrain.utils.logger import get_logger
from transformers import WhisperTokenizer


logger = get_logger(__name__)

SLU_TAG_PATTERN = re.compile(r"<(?P<label>[^<>/\s]+)>")


def setup_sf_tokenizer(
    tokenizer: WhisperTokenizer,
    csv_paths: list[str | Path],
    target_column: str = "tun_slu_annotation",
    save_path: str | Path | None = None,
) -> tuple[WhisperTokenizer, list[int]]:
    """Create or load a Whisper tokenizer with SLURP-TN SF tokens.

    Arguments
    ---------
    tokenizer
        Base Whisper tokenizer.
    csv_paths
        CSV manifest paths used to infer the slot-filling concept labels.
    target_column
        Column containing the SLU annotations.
    save_path
        Optional path where the customized tokenizer is saved.

    Returns
    -------
    tokenizer
        Tokenizer with slot-filling tokens added.
    sf_token_ids
        Token IDs corresponding to the slot-filling tokens.
    """
    save_path = Path(save_path) if save_path is not None else None

    concepts = infer_concepts_from_csvs(
        csv_paths=csv_paths,
        target_column=target_column,
    )
    sf_tokens = build_sf_tokens(concepts)

    if save_path is not None and tokenizer_exists(save_path):
        logger.info("Loading existing SF tokenizer from: %s", save_path)
        tokenizer = tokenizer.__class__.from_pretrained(save_path)
    else:
        add_tokens_to_tokenizer(tokenizer=tokenizer, sf_tokens=sf_tokens)

        if save_path is not None:
            save_path.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(save_path)
            logger.info("Tokenizer saved to: %s", save_path)

    validate_sf_tokens(tokenizer=tokenizer, sf_tokens=sf_tokens)

    sf_token_ids = tokenizer.convert_tokens_to_ids(sf_tokens)

    logger.info("Concept labels: %s", concepts)
    logger.info("SF tokens: %s", sf_tokens)
    logger.info("SF token IDs: %s", sf_token_ids)
    logger.info("Tokenizer vocabulary size: %d", len(tokenizer))

    return tokenizer, sf_token_ids


def tokenizer_exists(save_path: str | Path) -> bool:
    """Return True if a saved tokenizer seems to exist."""
    save_path = Path(save_path)

    if not save_path.exists():
        return False

    expected_files = [
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]

    return any((save_path / file_name).exists() for file_name in expected_files)


def add_tokens_to_tokenizer(
    tokenizer: WhisperTokenizer,
    sf_tokens: list[str],
) -> None:
    """Add missing slot-filling tokens to a Whisper tokenizer."""
    existing_tokens = set(tokenizer.get_vocab().keys())
    new_tokens = [
        token for token in sf_tokens
        if token not in existing_tokens
    ]

    if not new_tokens:
        logger.info("All SF tokens are already present in the tokenizer.")
        return

    logger.info("Adding %d SF tokens to Whisper tokenizer.", len(new_tokens))

    tokenizer.add_special_tokens(
        {"additional_special_tokens": new_tokens}
    )


def infer_concepts_from_csvs(
    csv_paths: list[str | Path],
    target_column: str,
) -> list[str]:
    """Infer concept labels from SLU annotation columns in CSV manifests."""
    concepts: set[str] = set()

    for csv_path in csv_paths:
        csv_path = Path(csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV manifest not found: {csv_path}")

        with open(csv_path, encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)

            if reader.fieldnames is None:
                raise ValueError(f"CSV manifest has no header: {csv_path}")

            if target_column not in reader.fieldnames:
                raise ValueError(
                    f"Column '{target_column}' not found in {csv_path}. "
                    f"Available columns are: {reader.fieldnames}"
                )

            for row in reader:
                annotation = row[target_column]
                concepts.update(extract_concepts(annotation))

    if not concepts:
        raise ValueError(
            "No SLU concept labels were found. "
            "Check that the CSVs contain annotations like '<time> ... >'."
        )

    return sorted(concepts)


def extract_concepts(annotation: str) -> list[str]:
    """Extract concept labels from one SLU annotation string.

    Example
    -------
    'بلاهي فيقني <time> السبعة و النصف >'
    returns:
    ['time']
    """
    return [
        match.group("label").strip()
        for match in SLU_TAG_PATTERN.finditer(annotation)
    ]


def build_sf_tokens(concepts: list[str]) -> list[str]:
    """Build tokenizer special tokens from concept labels."""
    concept_tokens = [f"<{concept}>" for concept in concepts]
    return concept_tokens + [">"]


def validate_sf_tokens(
    tokenizer: WhisperTokenizer,
    sf_tokens: list[str],
) -> None:
    """Ensure all expected SF tokens exist in the tokenizer."""
    vocab = tokenizer.get_vocab()

    missing_tokens = [
        token for token in sf_tokens
        if token not in vocab
    ]

    if missing_tokens:
        raise ValueError(
            "The following SF tokens are missing from the tokenizer: "
            f"{missing_tokens}. If you loaded an old tokenizer, delete "
            "custom_tokenizer_path and rerun."
        )