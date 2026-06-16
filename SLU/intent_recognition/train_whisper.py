"""Whisper intent recognition fine-tuning for SLURP-TN.

End-to-end training for spoken intent recognition.

Authors
-------
 * Haroun Elleuch, 2026
"""

import logging
import os
import sys

import speechbrain as sb
import torch
from hyperpyyaml import load_hyperpyyaml
import sys
from pathlib import Path
import logging


import speechbrain as sb
import speechbrain.core
from tqdm.auto import tqdm as auto_tqdm

import torch
from hyperpyyaml import load_hyperpyyaml

from speechbrain.utils.logger import get_logger
from speechbrain.utils.distributed import run_on_main


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from evaluation.intent_recognition import (generate_metrics_report, make_dir,
                                           save_predictions_to_jsonl)

speechbrain.core.tqdm = auto_tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
    force=True,
)

logger = get_logger(__name__)


class IR(sb.Brain):

    def compute_forward(self, batch, stage):
        """Compute intent predictions from a batch of audio waveforms."""

        batch = batch.to(self.device)

        # `wavs` contains the padded waveforms:
        #     shape: [batch_size, max_num_samples]
        #
        # `lens` contains the relative length of each waveform before padding.
        # For example, a value of 0.5 means that only the first 50% of the
        # corresponding row in `wavs` contains real audio.
        wavs, lens = batch.sig

        # Extract frame-level speech representations with the Whisper encoder.
        #
        # Since the model is configured with `encoder_only=True`, the output has
        # approximately the following shape:
        #     [batch_size, num_encoder_frames, hidden_size]
        embeddings = self.modules.whisper(wavs)

        # Convert SpeechBrain's relative waveform lengths back to absolute lengths
        # expressed in audio samples.
        #
        # Example:
        #     wavs.shape[1] = 160000 samples
        #     lens = 0.5
        #     absolute_lens = 80000 valid samples
        absolute_lens = lens * wavs.shape[1]

        # Whisper processes audio in a fixed 30-second input window.
        #
        # At 16 kHz:
        #     30 seconds × 16000 samples/second = 480000 samples
        #
        # StatisticsPooling expects relative sequence lengths between 0 and 1.
        # Dividing by 480000 therefore indicates which fraction of the Whisper
        # encoder output corresponds to real audio rather than padding.
        #
        # `clamp(max=1.0)` ensures that the relative length never exceeds 1.
        encoder_lens = (absolute_lens / 480000).clamp(max=1.0)

        # Average the Whisper encoder representations over the valid audio frames.
        #
        # Padding frames are ignored using `encoder_lens`.
        # StatisticsPooling returns:
        #     [batch_size, 1, hidden_size]
        #
        # After `squeeze(1)`:
        #     [batch_size, hidden_size]
        pooled_embeddings = self.hparams.avg_pool(
            x=embeddings,
            lengths=encoder_lens,
        ).squeeze(1)

        # Map each utterance-level embedding to one score per intent class.
        #
        # Output shape:
        #     [batch_size, number_of_intents]
        outputs = self.modules.output_mlp(pooled_embeddings)

        # Convert the classifier scores into log-probabilities for the NLL loss.
        outputs = self.hparams.log_softmax(outputs)

        # Return the intent predictions and original waveform lengths.
        # The lengths are later passed to the evaluation metric.
        return outputs, lens

    def compute_objectives(self, inputs, batch, stage):
        """Computes the loss given the predicted and targeted outputs.

        Arguments
        ---------
        inputs : tensors
            The output tensors from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """

        predictions, lens = inputs
        targets = batch.intent_encoded.data

        loss = self.hparams.compute_cost(predictions, targets.squeeze(1))

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(
                batch.id,
                predictions,
                targets,
                lens,
            )

            scores, indexes = torch.max(predictions, dim=-1)
            predicted_labels = self.intent_encoder.decode_torch(indexes)
            target_labels = [
                self.intent_encoder.decode_torch(target)[0]
                for target in targets
            ]

            prediction_dict = {
                "ids": [str(utt_id) for utt_id in batch.id],
                "scores": [round(score.exp().item(), 5) for score in scores],
                "indexes": [index.item() for index in indexes],
                "predicted_labels": list(predicted_labels),
                "target_labels": target_labels,
            }
            self.prediction_log.append(prediction_dict)
            
        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )
        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()
            self.prediction_log = []

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                "error": self.error_metrics.summarize("average"),
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stats["error"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            (
                old_lr_whisper,
                new_lr_whisper,
            ) = self.hparams.lr_annealing_whisper(stats["error"])
            sb.nnet.schedulers.update_learning_rate(
                self.whisper_optimizer, new_lr_whisper
            )

            # Runs on main process only
            save_predictions_to_jsonl(
                metrics_log_dir=self.hparams.metrics_log_dir,
                filename=f"classifications_epoch_{epoch}.jsonl",
                predictions=self.prediction_log,
            )

            # Runs on main process only
            classification_scores = generate_metrics_report(
                predictions=self.prediction_log,
                plot_matrix=True,
                plot_path=os.path.join(
                    self.hparams.metrics_log_dir, f"confusion_matrix_epoch_{epoch}.png"),
                report_path=os.path.join(
                    self.hparams.metrics_log_dir, f"metrics_report_epoch_{epoch}.json"),
                verbose=False,
                return_dict=True
            )

            # If main process, add classification scores to stats:
            if classification_scores:
                stats.update(classification_scores)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr, "whisper_lr": old_lr_whisper},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_and_keep_only(
                meta=stats,
                max_keys=["macro_f1", "weighted_f1"],
                num_to_keep=2,
                name=f"epoch_{epoch}")

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

            save_predictions_to_jsonl(
                metrics_log_dir=self.hparams.metrics_log_dir,
                filename="classifications_test.jsonl",
                predictions=self.prediction_log,
            )
            generate_metrics_report(
                predictions=self.prediction_log,
                plot_matrix=True,
                plot_path=os.path.join(
                    self.hparams.metrics_log_dir, "confusion_matrix_test.png"),
                report_path=os.path.join(
                    self.hparams.metrics_log_dir, "metrics_report_test.json"),
                verbose=True,
                return_dict=False,
            )

    def init_optimizers(self):
        "Initializes the whisper optimizer and model optimizer"
        self.whisper_optimizer = self.hparams.whisper_opt_class(
            self.modules.whisper.parameters()
        )
        self.optimizer = self.hparams.opt_class(
            self.hparams.model.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "whisper_opt", self.whisper_optimizer
            )
            self.checkpointer.add_recoverable("optimizer", self.optimizer)

        self.optimizers_dict = {
            "model_optimizer": self.optimizer,
            "whisper_optimizer": self.whisper_optimizer,
        }


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    We expect `prepare_common_intent` to have been called before this,
    so that the `train.csv`, `dev.csv`,  and `test.csv` manifest files
    are available.

    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "dev" that correspond
        to the appropriate DynamicItemDataset object.
    """

    # Initialization of the label encoder. The label encoder assigns to each
    # of the observed label a unique index (e.g, 'dial01': 0, 'dial02': 1, ..)
    intent_encoder = sb.dataio.encoder.CategoricalEncoder()
    intent_encoder.expect_len(hparams["n_intents"])

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load and optionally resample audio."""
        return sb.dataio.dataio.read_audio(wav)

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("intent")
    @sb.utils.data_pipeline.provides("intent", "intent_encoded")
    def label_pipeline(intent):
        yield intent
        intent_encoded = intent_encoder.encode_label_torch(intent)
        yield intent_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    for dataset in ["train", "dev", "test"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams[f"{dataset}_csv"],
            replacements=None,
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "intent_encoded"],
        )

    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mapping.
    intent_encoder_file = os.path.join(
        hparams["save_folder"], "intent_encoder.txt"
    )
    intent_encoder.load_or_create(
        path=intent_encoder_file,
        from_didatasets=[
            datasets["train"],
            datasets["dev"],
            datasets["test"],
        ],
        output_key="intent",
    )

    return datasets, intent_encoder


# Recipe begins!
if __name__ == "__main__":
    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(  # metrics dir
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    make_dir(hparams["metrics_log_dir"])

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

    # Create dataset objects "train", "dev", and "test" and intent_encoder
    datasets, intent_encoder = dataio_prep(hparams)

    # Initialize the Brain object to prepare for mask training.
    ir_brain = IR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    ir_brain.intent_encoder = intent_encoder

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    ir_brain.fit(
        epoch_counter=ir_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["dev"],
        train_loader_kwargs=hparams["train_dataloader_options"],
        valid_loader_kwargs=hparams["test_dataloader_options"],
    )

    # Load the best checkpoint for evaluation
    test_stats = ir_brain.evaluate(
        test_set=datasets["test"],
        min_key="error",
        test_loader_kwargs=hparams["test_dataloader_options"],
    )
