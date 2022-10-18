from whisper.model import Whisper
import os
from loss import get_loss_single_segment as get_loss_function
from typing import Dict, List, Tuple, Iterable, Optional, Sequence, Union, TYPE_CHECKING
import torch
from whisper import _MODELS, _download, ModelDimensions, available_models
from whisper.tokenizer import Tokenizer, get_tokenizer
from typing import Tuple
import numpy as np

def detect_language_with_gradients(model: "Whisper", mel: torch.Tensor, tokenizer: Tokenizer = None) -> Tuple[torch.Tensor, List[dict]]:
    """
    Detect the spoken language in the audio, and return them as list of strings, along with the ids
    of the most probable language tokens and the probability distribution over all language tokens.
    This is performed outside the main decode loop in order to not interfere with kv-caching.

    Returns
    -------
    language_tokens : torch.Tensor, shape = (n_audio,)
        ids of the most probable language tokens, which appears after the startoftranscript token.
    language_probs : List[Dict[str, float]], length = n_audio
        list of dictionaries containing the probability distribution over all languages.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer(model.is_multilingual)
    if tokenizer.language is None or tokenizer.language_token not in tokenizer.sot_sequence:
        raise ValueError(f"This model doesn't have language tokens so it can't perform lang id")

    single = mel.ndim == 2
    if single:
        mel = mel.unsqueeze(0)

    # skip encoder forward pass if already-encoded audio features were given
    if mel.shape[-2:] != (model.dims.n_audio_ctx, model.dims.n_audio_state):
        mel = model.encoder(mel)

    # forward pass using a single token, startoftranscript
    n_audio = mel.shape[0]
    x = torch.tensor([[tokenizer.sot]] * n_audio).to(mel.device)  # [n_audio, 1]
    logits = model.logits(x, mel)[:, 0]

    # collect detected languages; suppress all non-language tokens
    mask = torch.ones(logits.shape[-1], dtype=torch.bool)
    mask[list(tokenizer.all_language_tokens)] = False
    logits[:, mask] = -np.inf
    language_tokens = logits.argmax(dim=-1)
    language_token_probs = logits.softmax(dim=-1).cpu()
    language_probs = [
        {
            c: language_token_probs[i, j].item()
            for j, c in zip(tokenizer.all_language_tokens, tokenizer.all_language_codes)
        }
        for i in range(n_audio)
    ]

    if single:
        language_tokens = language_tokens[0]
        language_probs = language_probs[0]

    return language_tokens, language_probs, logits


class WhisperWithGradient(Whisper):
    loss = get_loss_function



def load_model_with_gradients(name: str, device: Optional[Union[str, torch.device]] = None, download_root: str = None, in_memory: bool = False, with_grad = True) -> Whisper:
    """
    Load a Whisper ASR model
    Parameters
    ----------
    name : str
        one of the official model names listed by `whisper.available_models()`, or
        path to a model checkpoint containing the model dimensions and the model state_dict.
    device : Union[str, torch.device]
        the PyTorch device to put the model into
    download_root: str
        path to download the model files; by default, it uses "~/.cache/whisper"
    in_memory: bool
        whether to preload the model weights into host memory
    Returns
    -------
    model : Whisper
        The Whisper ASR model instance
    """

    if download_root is None:
        download_root = os.getenv(
            "XDG_CACHE_HOME", 
            os.path.join(os.path.expanduser("~"), ".cache", "whisper")
        )

    if name in _MODELS:
        checkpoint_file = _download(_MODELS[name], download_root, in_memory)
    elif os.path.isfile(name):
        checkpoint_file = open(name, "rb").read() if in_memory else name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    with (io.BytesIO(checkpoint_file) if in_memory else open(checkpoint_file, "rb")) as fp:
        checkpoint = torch.load(fp, map_location=device)
    del checkpoint_file

    dims = ModelDimensions(**checkpoint["dims"])
    model = WhisperWithGradient(dims) if with_grad else Whisper(dims)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model.to(device)




class WhisperWrapper(torch.nn.Module):
    def __init__(self, name: str, **kwargs):
        super(WhisperWrapper,self).__init__()
        self.model = load_model_with_gradients(name,**kwargs)