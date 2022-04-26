#!/usr/bin/env python
import gft
import argparse, os, paddle, sys

parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--device', type=str, default=paddle.get_device())
parser.add_argument('--lang', type=str, help="en|zh", default='en')
parser.add_argument('--model', type=str, help="conformer_wenetspeech|transformer_librispeech", default='transformer_librispeech')
args = parser.parse_args()

if __name__ == '__main__':
    from paddlespeech.cli import ASRExecutor, TextExecutor
    asr_executor = ASRExecutor()
    text_executor = TextExecutor()

    for line in sys.stdin:
        wavfile = line.rstrip()

        text = asr_executor(
            audio_file = os.path.abspath(os.path.expanduser(wavfile)),
            lang=args.lang,
            model=args.model,
            device=args.device)

        if args.lang == 'zh':
            text_with_punc = text_executor(
                text=text,
                lang=args.lang,
                model=args.model,
                task='punc',
                device=args.device)
            print('\t'.join([wavfile, text, text_with_punc]))
        else:
            print(wavfile + '\t' + text)        

    
