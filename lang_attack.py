from robust_speech.adversarial.attacks.pgd import SNRPGDAttack
from sb_whisper_binding import WhisperASR
from whisper_with_gradients import detect_language_with_gradients
from whisper.audio import CHUNK_LENGTH, SAMPLE_RATE, N_FRAMES, HOP_LENGTH, pad_or_trim, log_mel_spectrogram
import robust_speech as rs
import torch.nn as nn
import torch
from robust_speech.adversarial.utils import TargetGenerator
import copy 

def compute_forward_lang(whisper_asr_brain, batch, stage):
    assert stage == rs.Stage.ATTACK
    batch = batch.to(whisper_asr_brain.device)
    wavs, wav_lens = batch.sig
    tokens_bos, _ = batch.tokens_bos

    if hasattr(whisper_asr_brain.hparams, "smoothing") and whisper_asr_brain.hparams.smoothing:
        wavs = whisper_asr_brain.hparams.smoothing(wavs, wav_lens)

    if hasattr(whisper_asr_brain.modules, "env_corrupt"):
        wavs_noise = whisper_asr_brain.modules.env_corrupt(wavs, wav_lens)
        wavs = torch.cat([wavs, wavs_noise], dim=0)
        wav_lens = torch.cat([wav_lens, wav_lens])
        tokens_bos = torch.cat([tokens_bos, tokens_bos], dim=0)

    if hasattr(whisper_asr_brain.hparams, "augmentation"):
        wavs = whisper_asr_brain.hparams.augmentation(wavs, wav_lens)
    # Forward pass
    tokens, _ = batch.tokens
    audio = wavs[0]
    mel = log_mel_spectrogram(audio)
    mel = pad_or_trim(mel,N_FRAMES)
    language_tokens, language_probs, logits = detect_language_with_gradients(
        whisper_asr_brain.modules.whisper.model,mel
    )
    print(language_probs["en"], language_probs["sd"])
    return language_tokens, language_probs, logits

def compute_objectives_lang(
        whisper_asr_brain, lang_token, predictions, batch, stage, reduction="none",**kwargs
    ):
    assert stage == rs.Stage.ATTACK
    language_tokens, language_probs, logits = predictions
    tokens=lang_token.to(whisper_asr_brain.device)
    loss_fct = nn.CrossEntropyLoss(reduction=reduction)
    loss = loss_fct(logits,tokens)
    return loss

class WhisperLangID(WhisperASR):
    def __init__(self,asr_brain,lang_token):
        assert isinstance(asr_brain,WhisperASR)
        self.asr_brain=asr_brain
        self.device = self.asr_brain.device
        self.modules = self.asr_brain.modules
        self.lang_token = lang_token

    def compute_forward(self,*args,**kwargs):
        return compute_forward_lang(self.asr_brain,*args,**kwargs)
    def compute_objectives(self,*args,**kwargs):
        return compute_objectives_lang(self.asr_brain,self.lang_token,*args,**kwargs)

class WhisperLanguageAttack(SNRPGDAttack):
    def __init__(self,asr_brain,*args,language="es",targeted_for_language=True,**kwargs):
        self.language = "<|"+language.strip("<|>")+"|>"
        self.lang_token = torch.LongTensor(asr_brain.tokenizer.encode(self.language))
        super(WhisperLanguageAttack,self).__init__(WhisperLangID(asr_brain,self.lang_token),*args,targeted=targeted_for_language,**kwargs)

