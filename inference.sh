#! /bin/bash
python inference.py --model tiny
python inference.py --model tiny.en

python inference.py --model base.en --config untargeted-35 --split whisper.base.en
python inference.py --model small --config untargeted-40 --split whisper.small
python inference.py --model tiny --config untargeted-40 --split whisper.small
python inference.py --model tiny --config untargeted-40 --split original

python inference.py --model medium --config language-lithuanian --split lithuanian.tagalog
python inference.py --model medium --config language-indonesian --split original
python inference.py --model medium --config language-english --split english.serbian