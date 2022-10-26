# whisper_attack

This repository contains code to fool Whisper ASR models with adversarial examples. It accompanies a paper currently under submission.

We provide code to generate examples as we did, and to evaluate Whisper on our examples via huggingface transformers.

## Requirements

Install [robust_speech](https://github.com/RaphaelOlivier/robust_speech) and [whisper](https://github.com/openai/whisper)

To use the HF inference pipeline you'll need `transformers>=4.23.0`, `datasets>=2.5.0` and `evaluate>=0.2.2`.

## Usage

## Generate adversarial examples
The `run_attack.py` file runs the robust_speech attack evaluation script. Configuration files in `attack_configs/` detail the attacks, datasets used and hyperparameters, and can be customized with command line arguments. Model configurations in `model_configs/` detail the loading information for each Whisper model.

For examples, please check our bash scripts which reproduce the attacks we ran in the paper:
* `cw.sh` runs a targeted attack on the ASR decoder
* `pgd.sh` runs an untargeted attack on the ASR decoder (with 35dB and 40dB SNR respectively)
* `smooth.sh` runs an untargeted attack on the ASR decoder while using the Randomized Smoothing defense
* `lang.sh` runs a targeted attack on the language detector, leading to a degradation of ASR performance. We run it with 7 source languages and 3 target languages.
* `rand.sh` applies gaussian noise for comparison

You will need to setup the datasets for robust_speech. For the language detection attack those are CommonVoice datasets in the source languages. For all other attacks, it's the LibriSpeech test-clean set. If like use you generate attacks on a subset of the dataset, you should generate the subset csvs. Here is an example for LibriSpeech:
```head test-clean.csv -n 101 > test-clean-100.csv```

## Use our precomputed adversarial examples

in the `whisper_adversarial_examples` folder, we provide all our precomputed adversarial examples in the form of a Huggingface dataset. You can use that dataset directly with the `inference.py` script, for example: 
```
python inference.py --model whisper-medium.en --config untargeted-35
```
More examples are proposed in the `inference.sh` script.

The dataset is also available [on the Hub](https://huggingface.co/datasets/RaphaelOlivier/whisper_adversarial_examples). Or you can just download the [archives](https://data.mendeley.com/datasets/96dh52hz9r/draft?a=ee30841f-1832-41ec-bdac-bf3e5b67073c).
