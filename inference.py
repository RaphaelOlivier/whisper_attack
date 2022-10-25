from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import torch
from evaluate import load
from whisper_feature_extractor_with_grad import WhisperFeatureExtractorWithGrad
# load model and processor

model_name = "medium"
hub_path = "openai/whisper-"+model_name
processor = WhisperProcessor.from_pretrained(hub_path)
model = WhisperForConditionalGeneration.from_pretrained(hub_path).to("cuda")

# load dummy dataset and read soundfiles

librispeech_eval = load_dataset("librispeech_asr", "clean", split="test")
librispeech_short = librispeech_eval.filter(lambda example: example["audio"]["array"].shape[0]<80000)

def map_to_pred(batch):
    input_features = processor(batch["audio"][0]["array"], return_tensors="pt").input_features
    #sos = torch.tensor([[50259,50359,50363]])
    #sos = processor.get_decoder_prompt_ids(language = "en", task = "transcribe")
    #logits = model(input_features.to("cuda"),decoder_input_ids = sos.to("cuda")).logits
    #predicted_ids = torch.argmax(logits, dim=-1)
    predicted_ids = model.generate(input_features.to("cuda"))
    transcription = processor.batch_decode(predicted_ids, normalize = True)
    batch['text'][0] = processor.tokenizer._normalize(batch['text'][0])
    batch["transcription"] = transcription
    print(batch['text'],batch["transcription"])
    return batch

result = librispeech_short.map(map_to_pred, batched=True, batch_size=1)

wer = load("wer")

print(wer.compute(predictions=result["text"], references=result["transcription"]))
