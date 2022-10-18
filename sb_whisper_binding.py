
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

        options = {
            "language": self.hparams.language if hasattr(self.hparams, "language") else None,
            "fp16": self.hparams.fp16 if hasattr(self.hparams, "fp16") else False,
            "without_timestamps": self.hparams.without_timestamps if hasattr(self.hparams, "without_timestamps") else True,
            "beam_size": self.hparams.beam_size if hasattr(self.hparams, "beam_size") else None
        }
        loss_options = { 
            "confidence": self.hparams.confidence if hasattr(self.hparams, "confidence") else 0.,
            "correct_first_word": self.hparams.correct_first_word if hasattr(self.hparams, "correct_first_word") else False
        }

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
                result = self.modules.whisper.model.loss(wavs[0],tokens[0], task="transcribe", **loss_options, **options)
                loss = result["loss"].detach()
                #logits = result["logits"]
                #pred_tokens = logits.argmax(dim=-1)
                result = self.modules.whisper.model.transcribe(wavs[0], task="transcribe", **options)
                text = result["text"]
                pred_tokens = torch.LongTensor([self.tokenizer.encode(text)])
        else:
            result = self.modules.whisper.model.loss(wavs[0],tokens[0], task="transcribe", **loss_options, **options)
            loss = result["loss"]
            #logits = self.modules.whisper.model.transcribe(wavs[0], beam_size=1)
            logits = result["logits"]
            pred_tokens = logits.argmax(dim=-1)
        return loss, pred_tokens, wav_lens

    def get_tokens(self, predictions):
        #text = predictions[1]["text"]
        #tokens = torch.LongTensor([self.tokenizer.encode(text)])
        tokens = predictions[1][:,:-1].cpu()
        return tokens

    def compute_objectives(
        self, predictions, batch, stage, adv=False, targeted=False, reduction="mean"
    ):
        """Computes the loss (CTC+NLL) given predictions and targets."""
        
        loss, pred_tokens, wav_lens = predictions

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
            predicted_words = [self.tokenizer.decode(t).strip().upper().translate(str.maketrans('', '', string.punctuation)) for t in pred_tokens]
            predicted_words = [wrd.split(" ") for wrd in predicted_words]
            target_words = [wrd.upper().upper().translate(str.maketrans('', '', string.punctuation)).split(" ") for wrd in batch.wrd]

            if adv:
                if targeted:
                    self.adv_wer_metric_target.append(
                        ids, predicted_words, target_words
                    )
                    self.adv_cer_metric_target.append(
                        ids, predicted_words, target_words
                    )
                    self.adv_ser_metric_target.append(
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
            print(" ".join(predicted_words[0]))
        return loss

    def init_optimizers(self):
        "Initializes the optimizer and model optimizer"
        self.optimizer = self.hparams.opt_class(
            self.hparams.model.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("optimizer", self.optimizer)
