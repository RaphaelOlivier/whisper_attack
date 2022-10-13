
import logging
import os
import sys

import speechbrain as sb
import torch
import torch.nn as nn
import string
from whisper_with_gradients import WhisperWithGradient

import robust_speech as rs
from robust_speech.adversarial.brain import AdvASRBrain

logger = logging.getLogger(__name__)


# Define training procedure


class WhisperASR(AdvASRBrain):
    """
    Whisper ASR model
    """

    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        if not stage == rs.Stage.ATTACK:
            batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        # if self.filter is not None:
        #     wavs = self.filter(wavs)
        tokens_bos, _ = batch.tokens_bos
        # wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        # Add augmentation if specified

        if hasattr(self.hparams, "smoothing") and self.hparams.smoothing:
            wavs = self.hparams.smoothing(wavs, wav_lens)

        if stage == sb.Stage.TRAIN or stage == rs.Stage.ATTACK:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                tokens_bos = torch.cat([tokens_bos, tokens_bos], dim=0)

            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)
        # Forward pass
        tokens, _ = batch.tokens
        if stage != sb.Stage.TRAIN and stage != rs.Stage.ATTACK:
            # Decode token terms to words
            with torch.no_grad():
                loss = self.modules.whisper.model.loss(wavs[0],tokens[0]).detach()
                result_decoding = self.modules.whisper.model.transcribe(wavs[0], beam_size=1)
        else:
            loss = self.modules.whisper.model.loss(wavs[0],tokens[0])
            result_decoding=None
        return loss, result_decoding, wav_lens

    def get_tokens(self, predictions):
        text = predictions[1]["text"]
        tokens = torch.LongTensor(self.tokenizer.encode(text))
        return tokens

    def compute_objectives(
        self, predictions, batch, stage, adv=False, targeted=False, reduction="mean"
    ):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        loss, result_decoding, wav_lens = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens_eos = torch.cat([tokens_eos, tokens_eos], dim=0)
            tokens_eos_lens = torch.cat(
                [tokens_eos_lens, tokens_eos_lens], dim=0)
            tokens = torch.cat([tokens, tokens], dim=0)
            tokens_lens = torch.cat([tokens_lens, tokens_lens], dim=0)

        if stage != sb.Stage.TRAIN and stage != rs.Stage.ATTACK:
            # Decode token terms to words
            predicted_words = result_decoding["text"]
            predicted_words = predicted_words.upper().strip()
            predicted_words = predicted_words.translate(str.maketrans('', '', string.punctuation))
            predicted_words = [predicted_words.split(" ")]
            target_words = [wrd.split(" ") for wrd in batch.wrd]

            if adv:
                if targeted:
                    self.adv_wer_metric_target.append(
                        ids, predicted_words, target_words
                    )
                    self.adv_cer_metric_target.append(
                        ids, predicted_words, target_words
                    )
                else:
                    self.adv_wer_metric.append(
                        ids, predicted_words, target_words)
                    self.adv_cer_metric.append(
                        ids, predicted_words, target_words)
            else:
                self.wer_metric.append(ids, predicted_words, target_words)
                self.cer_metric.append(ids, predicted_words, target_words)
            # print(predicted_words, target_words)
        return loss

    def init_optimizers(self):
        "Initializes the optimizer and model optimizer"
        self.optimizer = self.hparams.opt_class(
            self.hparams.model.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("optimizer", self.optimizer)

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        # check if we need to switch optimizer
        # if so change the optimizer from Adam to SGD

        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)

        # normalize the loss by gradient_accumulation step
        (loss / self.hparams.gradient_accumulation).backward()

        if self.step % self.hparams.gradient_accumulation == 0:
            # gradient clipping & early stop if loss is not fini
            self.check_gradients(loss)

            for p in self.optimizer.param_groups[0]['params']:
                1
                #print(p.data.size(), p.grad)
            self.optimizer.step()
            self.optimizer.zero_grad()

            # anneal lr every update
            self.hparams.noam_annealing(self.optimizer)
        return loss.detach()
