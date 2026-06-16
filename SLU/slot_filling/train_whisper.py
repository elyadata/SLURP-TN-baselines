"""Whisper slot-filling fine-tuning for SLURP-TN.

End-to-end training for speech recognition with slot-filling annotations.

Authors
-------
 * Haroun Elleuch, 2026
"""

import sys
from pathlib import Path
import logging


import speechbrain as sb
import speechbrain.core
from tqdm.auto import tqdm as auto_tqdm

import soundfile as sf
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from setup_sf_tokenizer import setup_sf_tokenizer
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import if_main_process, run_on_main
from speechbrain.utils.logger import get_logger

speechbrain.core.tqdm = auto_tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.scoring import (  # noqa: E402
    clean_whisper_tokens,
    compute_metrics_from_jsonl,
    compute_slu_metrics,
    write_metrics_txt,
    write_predictions_jsonl,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
    force=True,
)

logger = get_logger(__name__)


class WhisperSF(sb.Brain):
    """SpeechBrain Brain class for Whisper slot-filling fine-tuning."""

    def __init__(self, *args, **kwargs):
        """Initialize the Brain class."""
        super().__init__(*args, **kwargs)
        self.prediction_store = []

    def compute_forward(self, batch, stage):
        """Run the forward pass."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        bos_tokens, bos_tokens_lens = batch.tokens_bos

        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)
            bos_tokens = self.hparams.wav_augment.replicate_labels(bos_tokens)
            bos_tokens_lens = self.hparams.wav_augment.replicate_labels(
                bos_tokens_lens
            )

        abs_tokens_lens = (bos_tokens_lens * bos_tokens.shape[1]).long()
        pad_mask = (
            torch.arange(abs_tokens_lens.max(), device=self.device)[None, :]
            < abs_tokens_lens[:, None]
        )
        bos_tokens[~pad_mask] = self.tokenizer.pad_token_id

        enc_out, logits, _ = self.modules.whisper(wavs, bos_tokens)
        log_probs = self.hparams.log_softmax(logits)

        hyps = None
        if stage == sb.Stage.VALID:
            hyps, _, _, _ = self.hparams.valid_search(
                enc_out.detach(), wav_lens
            )
        elif stage == sb.Stage.TEST:
            hyps, _, _, _ = self.hparams.test_search(
                enc_out.detach(), wav_lens
            )

        return log_probs, hyps, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """Compute the training loss and store predictions for evaluation."""
        log_probs, hyps, wav_lens = predictions
        batch = batch.to(self.device)

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos

        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            tokens_eos = self.hparams.wav_augment.replicate_labels(tokens_eos)
            tokens_eos_lens = self.hparams.wav_augment.replicate_labels(
                tokens_eos_lens
            )

        loss = self.hparams.nll_loss(
            log_probs,
            tokens_eos,
            length=tokens_eos_lens,
        )

        if stage != sb.Stage.TRAIN:
            tokens, tokens_lens = batch.tokens

            predicted_words = [
                self.tokenizer.decode(t, skip_special_tokens=False).strip()
                for t in hyps
            ]

            target_words = undo_padding(tokens, tokens_lens)
            target_words = self.tokenizer.batch_decode(
                target_words,
                skip_special_tokens=False,
            )

            hyp_words = [
                clean_whisper_tokens(prediction)
                for prediction in predicted_words
            ]
            ref_words = [
                clean_whisper_tokens(reference)
                for reference in target_words
            ]

            for utt_id, ref, hyp in zip(ids, ref_words, hyp_words):
                self.prediction_store.append(
                    {
                        "id": str(utt_id),
                        "ref": ref.strip(),
                        "hyp": hyp.strip(),
                    }
                )

        return loss

    def on_stage_start(self, stage, epoch):
        """Reset prediction storage at the start of validation/test stages."""
        if stage != sb.Stage.TRAIN:
            self.prediction_store = []

    def on_stage_end(self, stage, stage_loss, epoch):
        """Log losses and metrics at the end of each stage."""
        stage_stats = {"loss": stage_loss}

        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
            return

        stage_stats.update(compute_slu_metrics(self.prediction_store))

        if stage == sb.Stage.VALID:
            lr = self.hparams.lr_annealing_whisper.current_lr

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            self.checkpointer.save_and_keep_only(
                name=f"_epoch_{epoch}",
                meta={"WER": stage_stats["WER"]},
                min_keys=["WER"],
            )

        elif stage == sb.Stage.TEST:
            if if_main_process():
                scores_dir = Path(self.scores_dir)
                scores_dir.mkdir(parents=True, exist_ok=True)

                predictions_file = scores_dir / "predictions.jsonl"
                metrics_file = scores_dir / "metrics.txt"

                write_predictions_jsonl(
                    predictions_file=predictions_file,
                    predictions=self.prediction_store,
                )

                metrics = compute_metrics_from_jsonl(
                    predictions_file=predictions_file,
                    task=getattr(self.hparams, "task", "slu"),
                )

                write_metrics_txt(
                    metrics_file=metrics_file,
                    metrics=metrics,
                )

                stage_stats.update(metrics)

            self.hparams.train_logger.log_stats(
                stats_meta={
                    "Epoch loaded": self.hparams.epoch_counter.current
                },
                test_stats=stage_stats,
            )


def dataio_prepare(hparams, tokenizer):
    """Prepare SpeechBrain datasets and dynamic pipelines."""
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"]
    )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"]
    )
    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"]
    )

    datasets = [train_data, valid_data, test_data]

    target_column = hparams.get("target_column", "tun_slu_annotation")

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load and optionally resample audio."""
        info = sf.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)

        if info.samplerate != hparams["sample_rate"]:
            sig = torchaudio.transforms.Resample(
                info.samplerate,
                hparams["sample_rate"],
            )(sig)

        return sig

    @sb.utils.data_pipeline.takes(target_column)
    @sb.utils.data_pipeline.provides(
        "wrd",
        "tokens_list",
        "tokens_bos",
        "tokens_eos",
        "tokens",
    )
    def text_pipeline(wrd):
        """Tokenize the target text with slot-filling tags."""
        if hparams.get("normalized_transcripts", False):
            wrd = tokenizer.normalize(wrd)

        yield wrd

        tokens_list = tokenizer.encode(wrd, add_special_tokens=False)
        yield tokens_list

        tokens_list = tokenizer.build_inputs_with_special_tokens(tokens_list)

        tokens_bos = torch.LongTensor(tokens_list[:-1])
        yield tokens_bos

        tokens_eos = torch.LongTensor(tokens_list[1:])
        yield tokens_eos

        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    sb.dataio.dataset.set_output_keys(
        datasets,
        [
            "id",
            "sig",
            "wrd",
            "tokens_list",
            "tokens_bos",
            "tokens_eos",
            "tokens",
        ],
    )

    return train_data, valid_data, test_data


def patch_searcher_for_sf(searcher, default_length=448):
    """Patch the searcher to handle slot-filling tokens."""
    if not hasattr(searcher, "max_attn_tokens") or searcher.max_attn_tokens is None:
        searcher.max_attn_tokens = getattr(
            searcher, "sample_len", default_length)


if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    if not hparams.get("skip_prep", False):
        logger.info("Preparing data...")

        from data_prep import prepare_slurptn  # noqa: E402

        run_on_main(
            prepare_slurptn,
            kwargs={
                "data_folder": hparams["data_folder"],
            },
        )
    else:
        logger.info("Skipping data preparation.")

    logger.info("Initializing training...")

    whisper_sf_brain = WhisperSF(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        opt_class=hparams["whisper_opt_class"],
    )

    if "pretrainer" in hparams:
        logger.info("Loading pretrained Whisper model...")
        hparams["pretrainer"].collect_files()
        hparams["pretrainer"].load_collected()

    logger.info("Loading tokenizer and adding SF tokens...")

    tokenizer, _ = setup_sf_tokenizer(
        tokenizer=whisper_sf_brain.modules["whisper"].tokenizer,
        csv_paths=[
            hparams["train_csv"],
            hparams["valid_csv"],
            hparams["test_csv"],
        ],
        target_column=hparams["target_column"],
        save_path=hparams["custom_tokenizer_path"],
    )

    whisper_sf_brain.tokenizer = tokenizer
    whisper_sf_brain.modules["whisper"].model.resize_token_embeddings(
        len(whisper_sf_brain.tokenizer)
    )

    emb_size = (
        whisper_sf_brain.modules["whisper"]
        .model.get_input_embeddings()
        .weight.size(0)
    )
    tok_size = len(whisper_sf_brain.tokenizer)
    assert emb_size == tok_size, (
        f"Mismatch: embeddings={emb_size}, tokenizer={tok_size}"
    )

    logger.info("Preparing datasets...")

    train_data, valid_data, test_data = dataio_prepare(
        hparams,
        whisper_sf_brain.tokenizer,
    )

    if hasattr(whisper_sf_brain.hparams, "valid_search"):
        patch_searcher_for_sf(whisper_sf_brain.hparams.valid_search)

    if hasattr(whisper_sf_brain.hparams, "test_search"):
        patch_searcher_for_sf(whisper_sf_brain.hparams.test_search)

    logger.info("Starting training...")

    whisper_sf_brain.fit(
        whisper_sf_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        progressbar=True,
        train_loader_kwargs=hparams["train_loader_kwargs"],
        valid_loader_kwargs=hparams["valid_loader_kwargs"],
    )

    if if_main_process():
        Path(whisper_sf_brain.hparams.dev_scores_dir).mkdir(
            parents=True,
            exist_ok=True,
        )
        Path(whisper_sf_brain.hparams.test_scores_dir).mkdir(
            parents=True,
            exist_ok=True,
        )

    logger.info("Evaluation on validation set...")

    whisper_sf_brain.scores_dir = whisper_sf_brain.hparams.dev_scores_dir
    whisper_sf_brain.evaluate(
        valid_data,
        test_loader_kwargs=hparams["test_loader_kwargs"],
        min_key="CoER",
        progressbar=True,
    )

    logger.info("Evaluation on test set...")

    whisper_sf_brain.scores_dir = whisper_sf_brain.hparams.test_scores_dir
    whisper_sf_brain.evaluate(
        test_data,
        test_loader_kwargs=hparams["test_loader_kwargs"],
        min_key="CoER",
        progressbar=True,
    )
