import whisper_with_gradients
from whisper.audio import load_audio
import torch
path_to_nat = "/home/rolivier/workhorse1/robust_speech/attacks/cw/wav2vec2-base-960h/1001/save/8455-210777-0040_nat.wav"
path_to_adv = "/home/rolivier/workhorse1/robust_speech/attacks/cw/wav2vec2-base-960h/1001/save/8455-210777-0040_adv.wav"
model = whisper_with_gradients.load_model_with_gradients("tiny")
result = model.transcribe(path_to_nat, beam_size=1)
print(result["text"])
transcription = result["text"]
# Then," said Sir Ferdando. There's nothing for it, but that we must take you with him.
random_transcription = "Here is a random transcription, unrelated to the input"

audio = torch.tensor(load_audio(path_to_nat), device=model.device,requires_grad = True)
og_audio = torch.clone(audio).detach()
for k in range(1000):
    loss = model.loss(audio,random_transcription) + 0.1*((audio-og_audio)*(audio-og_audio)).sum()
    print(loss)
    loss.backward()
    audio = (audio - 0.001*audio.grad).detach()
    audio.grad=None
    audio.requires_grad=True

result = model.transcribe(audio, beam_size=1)
print(result["text"])