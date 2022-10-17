from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Optional, Sequence, Union, TYPE_CHECKING
import tqdm
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical
from whisper.audio import CHUNK_LENGTH, SAMPLE_RATE, N_FRAMES, HOP_LENGTH, pad_or_trim, log_mel_spectrogram
from whisper.tokenizer import Tokenizer, get_tokenizer
from whisper.utils import compression_ratio, exact_div, format_timestamp, optional_int, optional_float, str2bool, write_txt, write_vtt, write_srt
from whisper.decoding import DecodingTask, DecodingOptions
if TYPE_CHECKING:
    from whisper.model import Whisper

@dataclass(frozen=True)
class LossResult:
    audio_features: Tensor
    language: str
    loss: Tensor
    logits: Tensor
    language_probs: Optional[Dict[str, float]] = None
    tokens: List[int] = field(default_factory=list)
    text: str = ""
    avg_logprob: float = np.nan
    no_speech_prob: float = np.nan
    temperature: float = np.nan
    compression_ratio: float = np.nan

class LossTask(DecodingTask):
    def _decoder_forward(self, audio_features: Tensor, tokens: Tensor, init_tokens_length: int):
        self.inference.initial_token_length = tokens.shape[-1]
        assert audio_features.shape[0] == tokens.shape[0]
        n_batch = tokens.shape[0]
        sum_logprobs: Tensor = torch.zeros(n_batch, device=audio_features.device)
        no_speech_probs = [np.nan] * n_batch
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        logits = self.inference.logits(tokens[:,:-1], audio_features)
        loss = loss_fct(logits[:,(init_tokens_length-1):].transpose(1,2),tokens[:,init_tokens_length:]).mean(dim=1)
        
        corrective_first_word_loss = loss_fct(logits[:,(init_tokens_length-1)],tokens[:,init_tokens_length]) #- loss_fct(logits[:,(init_tokens_length-1)],first_word_pred)
        corrective_first_word_loss = corrective_first_word_loss/tokens.size(1)
        loss=loss+corrective_first_word_loss

        self.inference.cleanup_caching()
        return loss, logits[:,(init_tokens_length-1):], no_speech_probs, sum_logprobs

    def run(self, mel: Tensor, label: Union[str, torch.Tensor]) -> List[LossResult]:
        self.decoder.reset()
        tokenizer: Tokenizer = self.tokenizer
        n_audio: int = mel.shape[0]
        audio_features: Tensor = self._get_audio_features(mel)  # encoder forward pass
        init_tokens: Tensor = torch.tensor([self.initial_tokens]).repeat(n_audio, 1).to(audio_features.device)
        init_tokens_length = init_tokens.size(-1)
        if isinstance(label,str):
            label = torch.tensor([tokenizer.encode(label)])
            
        input_tokens: Tensor = torch.tensor(label).repeat(n_audio, 1).to(audio_features.device)
        eos_tokens: Tensor = torch.tensor([[tokenizer.tokenizer.eos_token_id]]).repeat(n_audio, 1).to(audio_features.device)
        tokens = torch.cat([init_tokens,input_tokens,eos_tokens],dim=-1)
        # detect language if requested, overwriting the language token
        languages, language_probs = self._detect_language(audio_features, tokens)

        # repeat the audio & text tensors by the group size, for beam search or best-of-n sampling
        audio_features = audio_features.repeat_interleave(self.n_group, dim=0)
        tokens = tokens.repeat_interleave(self.n_group, dim=0).to(audio_features.device)

        # call the main sampling loop
        loss, logits, no_speech_probs, sum_logprobs = self._decoder_forward(audio_features, tokens, init_tokens_length)
        # reshape the tensors to have (n_audio, n_group) as the first two dimensions
        audio_features = audio_features[:: self.n_group]
        no_speech_probs = no_speech_probs[:: self.n_group]
        assert audio_features.shape[0] == len(no_speech_probs) == n_audio

        tokens = tokens.reshape(n_audio, self.n_group, -1)
        sum_logprobs = sum_logprobs.reshape(n_audio, self.n_group)

        # get the final candidates for each group, and slice between the first sampled token and EOT
        tokens, sum_logprobs = self.decoder.finalize(tokens, sum_logprobs)
        tokens: List[List[Tensor]] = [
            [t[self.sample_begin : (t == tokenizer.eot).nonzero()[0, 0]] for t in s] for s in tokens
        ]

        # select the top-ranked sample in each group
        selected = self.sequence_ranker.rank(tokens, sum_logprobs)
        tokens: List[List[int]] = [t[i].tolist() for i, t in zip(selected, tokens)]
        texts: List[str] = [tokenizer.decode(t).strip() for t in tokens]

        sum_logprobs: List[float] = [lp[i] for i, lp in zip(selected, sum_logprobs)]
        avg_logprobs: List[float] = [lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)]

        fields = (texts, languages, tokens, audio_features, logits, loss)
        return [
            LossResult(
                audio_features=features,
                language=language,
                tokens=tokens,
                text=text,
                logits=logits,
                loss=loss,
            )
            for text, language, tokens, features, logits, loss in zip(*fields)
        ]


def get_loss_from_mel(model: "Whisper", mel: Tensor, label: Union[str, torch.Tensor], options: DecodingOptions = DecodingOptions()) -> Union[LossResult, List[LossResult]]:
    """
    Performs decoding of 30-second audio segment(s), provided as Mel spectrogram(s).
    Parameters
    ----------
    model: Whisper
        the Whisper model instance
    mel: torch.Tensor, shape = (80, 3000) or (*, 80, 3000)
        A tensor containing the Mel spectrogram(s)
    options: DecodingOptions
        A dataclass that contains all necessary options for decoding 30-second segments
    Returns
    -------
    result: Union[LossResult, List[LossResult]]
        The result(s) of decoding contained in `LossResult` dataclass instance(s)
    """
    single = mel.ndim == 2
    if single:
        mel = mel.unsqueeze(0)

    result = LossTask(model, options).run(mel,label)
    if single:
        result = result[0]

    return result


def get_loss(
    model: "Whisper",
    audio: Union[str, np.ndarray, torch.Tensor],
    label: Union[str, torch.Tensor],
    *,
    verbose: Optional[bool] = None,
    temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold: Optional[float] = 2.4,
    logprob_threshold: Optional[float] = -1.0,
    no_speech_threshold: Optional[float] = 0.6,
    condition_on_previous_text: bool = True,
    **decode_options,
):
    """
    Transcribe an audio file using Whisper
    Parameters
    ----------
    model: Whisper
        The Whisper model instance
    audio: Union[str, np.ndarray, torch.Tensor]
        The path to the audio file to open, or the audio waveform
    verbose: bool
        Whether to display the text being decoded to the console. If True, displays all the details,
        If False, displays minimal details. If None, does not display anything
    temperature: Union[float, Tuple[float, ...]]
        Temperature for sampling. It can be a tuple of temperatures, which will be successfully used
        upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.
    compression_ratio_threshold: float
        If the gzip compression ratio is above this value, treat as failed
    logprob_threshold: float
        If the average log probability over sampled tokens is below this value, treat as failed
    no_speech_threshold: float
        If the no_speech probability is higher than this value AND the average log probability
        over sampled tokens is below `logprob_threshold`, consider the segment as silent
    condition_on_previous_text: bool
        if True, the previous output of the model is provided as a prompt for the next window;
        disabling may make the text inconsistent across windows, but the model becomes less prone to
        getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.
    decode_options: dict
        Keyword arguments to construct `DecodingOptions` instances
    Returns
    -------
    A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
    the spoken language ("language"), which is detected when `decode_options["language"]` is None.
    """
    dtype = torch.float16 if decode_options.get("fp16", True) else torch.float32
    if model.device == torch.device("cpu"):
        if torch.cuda.is_available():
            warnings.warn("Running model on CPU when CUDA is available")
        if dtype == torch.float16:
            warnings.warn("FP16 is not supported on CPU; using FP32 instead")
            dtype = torch.float32

    if dtype == torch.float32:
        decode_options["fp16"] = False
    mel = log_mel_spectrogram(audio)

    decode_options["language"] = "en"
    language = decode_options["language"]
    task = decode_options.get("task", "transcribe")
    tokenizer = get_tokenizer(model.is_multilingual, language=language, task=task)

    seek = 0
    input_stride = exact_div(
        N_FRAMES, model.dims.n_audio_ctx
    )  # mel frames per output token: 2
    time_precision = (
        input_stride * HOP_LENGTH / SAMPLE_RATE
    )  # time per output token: 0.02 (seconds)
    all_tokens = []
    all_segments = []
    prompt_reset_since = 0

    initial_prompt = decode_options.pop("initial_prompt", None) or []
    if initial_prompt:
        initial_prompt = tokenizer.encode(" " + initial_prompt.strip())
        all_tokens.extend(initial_prompt)

    def add_segment(
        *, start: float, end: float, text_tokens: torch.Tensor, result: LossResult
    ):
        text = tokenizer.decode([token for token in text_tokens if token < tokenizer.eot])
        if len(text.strip()) == 0:  # skip empty text output
            return

        all_segments.append(
            {
                "id": len(all_segments),
                "seek": seek,
                "start": start,
                "end": end,
                "text": text,
                "tokens": result.tokens,
                "temperature": result.temperature,
                "avg_logprob": result.avg_logprob,
                "compression_ratio": result.compression_ratio,
                "no_speech_prob": result.no_speech_prob,
            }
        )
        if verbose:
            print(f"[{format_timestamp(start)} --> {format_timestamp(end)}] {text}")

    # show the progress bar when verbose is False (otherwise the transcribed text will be printed)
    num_frames = mel.shape[-1]
    previous_seek_value = seek

    with tqdm.tqdm(total=num_frames, unit='frames', disable=verbose is not False) as pbar:
        while seek < num_frames:
            timestamp_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
            segment = pad_or_trim(mel[:, seek:], N_FRAMES).to(model.device).to(dtype)
            segment_duration = segment.shape[-1] * HOP_LENGTH / SAMPLE_RATE

            decode_options["prompt"] = all_tokens[prompt_reset_since:]
            result: LossResult = get_loss_from_mel(model, segment, label, DecodingOptions(**decode_options))
            tokens = torch.tensor(result.tokens)

            if no_speech_threshold is not None:
                # no voice activity check
                should_skip = result.no_speech_prob > no_speech_threshold
                if logprob_threshold is not None and result.avg_logprob > logprob_threshold:
                    # don't skip if the logprob is high enough, despite the no_speech_prob
                    should_skip = False

                if should_skip:
                    seek += segment.shape[-1]  # fast-forward to the next segment boundary
                    continue

            timestamp_tokens: torch.Tensor = tokens.ge(tokenizer.timestamp_begin)
            consecutive = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0].add_(1)
            if len(consecutive) > 0:  # if the output contains two consecutive timestamp tokens
                last_slice = 0
                for current_slice in consecutive:
                    sliced_tokens = tokens[last_slice:current_slice]
                    start_timestamp_position = (
                        sliced_tokens[0].item() - tokenizer.timestamp_begin
                    )
                    end_timestamp_position = (
                        sliced_tokens[-1].item() - tokenizer.timestamp_begin
                    )
                    add_segment(
                        start=timestamp_offset + start_timestamp_position * time_precision,
                        end=timestamp_offset + end_timestamp_position * time_precision,
                        text_tokens=sliced_tokens[1:-1],
                        result=result,
                    )
                    last_slice = current_slice
                last_timestamp_position = (
                    tokens[last_slice - 1].item() - tokenizer.timestamp_begin
                )
                seek += last_timestamp_position * input_stride
                all_tokens.extend(tokens[: last_slice + 1].tolist())
            else:
                duration = segment_duration
                timestamps = tokens[timestamp_tokens.nonzero().flatten()]
                if len(timestamps) > 0 and timestamps[-1].item() != tokenizer.timestamp_begin:
                    # no consecutive timestamps but it has a timestamp; use the last one.
                    # single timestamp at the end means no speech after the last timestamp.
                    last_timestamp_position = timestamps[-1].item() - tokenizer.timestamp_begin
                    duration = last_timestamp_position * time_precision

                add_segment(
                    start=timestamp_offset,
                    end=timestamp_offset + duration,
                    text_tokens=tokens,
                    result=result,
                )

                seek += segment.shape[-1]
                all_tokens.extend(tokens.tolist())

            if not condition_on_previous_text or result.temperature > 0.5:
                # do not feed the prompt tokens if a high temperature was used
                prompt_reset_since = len(all_tokens)

            # update progress bar
            pbar.update(min(num_frames, seek) - previous_seek_value)
            previous_seek_value = seek
        loss = result.loss
    return dict(text=tokenizer.decode(all_tokens[len(initial_prompt):]), segments=all_segments, language=language, loss = loss)


def get_loss_single_segment(
    model: "Whisper",
    audio: torch.Tensor,
    label: Union[str, torch.Tensor],
    *,
    verbose: Optional[bool] = None,
    temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold: Optional[float] = 2.4,
    logprob_threshold: Optional[float] = -1.0,
    no_speech_threshold: Optional[float] = 0.6,
    condition_on_previous_text: bool = True,
    **decode_options,
):
    """
    Transcribe an audio file using Whisper
    Parameters
    ----------
    model: Whisper
        The Whisper model instance
    audio: torch.Tensor
        The audio waveform
    verbose: bool
        Whether to display the text being decoded to the console. If True, displays all the details,
        If False, displays minimal details. If None, does not display anything
    temperature: Union[float, Tuple[float, ...]]
        Temperature for sampling. It can be a tuple of temperatures, which will be successfully used
        upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.
    compression_ratio_threshold: float
        If the gzip compression ratio is above this value, treat as failed
    logprob_threshold: float
        If the average log probability over sampled tokens is below this value, treat as failed
    no_speech_threshold: float
        If the no_speech probability is higher than this value AND the average log probability
        over sampled tokens is below `logprob_threshold`, consider the segment as silent
    condition_on_previous_text: bool
        if True, the previous output of the model is provided as a prompt for the next window;
        disabling may make the text inconsistent across windows, but the model becomes less prone to
        getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.
    decode_options: dict
        Keyword arguments to construct `DecodingOptions` instances
    Returns
    -------
    A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
    the spoken language ("language"), which is detected when `decode_options["language"]` is None.
    """
    dtype = torch.float16 if decode_options.get("fp16", True) else torch.float32
    audio = audio.to(model.device)
    mel = log_mel_spectrogram(audio)
    mel = pad_or_trim(mel,N_FRAMES)
    decode_options["language"] = "en"
    language = decode_options["language"]
    task = decode_options.get("task", "transcribe")
    result: LossResult = get_loss_from_mel(model, mel, label, DecodingOptions(**decode_options))
    loss = result.loss
    if loss.nelement() >1:
        loss = loss.mean(dim=1)
    if loss.ndim==0:
        loss = loss.unsqueeze(0)
    logits = result.logits 
    if logits.ndim == 2:
        logits=logits.unsqueeze(0)
    return dict(loss=loss,logits=logits)
