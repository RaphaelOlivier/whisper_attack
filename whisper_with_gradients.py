from whisper.model import Whisper
import os
from loss import get_loss_single_segment as get_loss_function
from typing import Dict, List, Tuple, Iterable, Optional, Sequence, Union, TYPE_CHECKING
import torch
from whisper import _MODELS, _download, ModelDimensions, available_models
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