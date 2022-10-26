from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import torch
from evaluate import load
import argparse
import warnings
# load model and processor
_MODELS = [
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large",
]

_CONFIGS = [
    "targeted",
    "untargeted-35",
    "untargeted-40",
    "language-armenian",
    "language-lithuanian",
    "language-czech",
    "language-danish",
    "language-indonesian",
    "language-italian",
    "language-english",
]

language_short = {
    "armenian":"hy-AM",
    "lithuanian":"lt",
    "czech":"cs",
    "danish":"da",
    "indonesian":"id",
    "italian":"it",
    "english":"en",
}

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-m','--model', help='Whisper model to use', default="tiny")
parser.add_argument('-c','--config', help='Dataset config to use', default="targeted")
parser.add_argument('-s','--split', help='Dataset split to use for the given config', default=None)


def main(args):
    language=None
    assert args.model in _MODELS, "Model %s invalid"%args.model
    assert args.config in _CONFIGS, "Config %s invalid"%args.config
    if "language" in args.config:
        assert args.split is not None, "split is required with language configs"
        lang = args.config.split('-')[-1]
        q = args.split.split('.')
        assert (
            (len(q)==2 
            and q[0]==lang 
            and q[1] in ["english","tagalog","serbian"])
            or args.split == "original"
            ), "Split %s invalid with config %s"%(args.split,args.config)
        if lang != "english":
            language = language_short[lang]
    else:
        if args.split is None:
            args.split = "whisper."+args.model
        else:
            assert (
                (args.split[:8]=="whisper." and args.split[8:] in _MODELS)
                or args.split == "original"
            ), "Split %s invalid with config %s"%(args.split,args.config)
    evaluate_on_adv_examples(args.model,args.config,args.split,language=language)

# load dummy dataset and read soundfiles

def evaluate_on_adv_examples(model_name,config_name,split_name,language=None):

    hub_path = "openai/whisper-"+model_name
    processor = WhisperProcessor.from_pretrained(hub_path)
    model = WhisperForConditionalGeneration.from_pretrained(hub_path).to("cuda")
    if language is not None:
        warnings.warn("Whisper on Huggingface does not support language detection, and seems to default to english in presence of any language. We set the language token to %s, which affects results if using the language detection attack."%language)
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=language,task = "transcribe")

    dataset = load_dataset("whisper_adversarial_examples",config_name ,split=split_name)

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
        return batch

    result = dataset.map(map_to_pred, batched=True, batch_size=1)

    wer = load("wer")
    for t in zip(result["text"],result["transcription"]):
        print(t)
    print(wer.compute(predictions=result["text"], references=result["transcription"]))


if __name__=="__main__":
    args = parser.parse_args()
    main(args)
