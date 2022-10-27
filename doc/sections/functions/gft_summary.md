# gft (general fine-tuning): gft_summary

<h1>gft_summary</h1>

Summarize (almost) anything: datasets, models, tasks, etc.

<h2>Summarize a dataset from HuggingFace and PaddleHub/PaddleNLP</h2>

Both hubs (H and P) provide support for popular datasets such as <a href="https://gluebenchmark.com/">glue</a>.
The two versions are similar to one another, though there are some small differences in the names of the
columns and labels.

```sh
# summarize the qqp subtask of glue (from HuggingFace)
gft_summary --data H:glue,qqp 2>/dev/null
```
<pre>
dataset: glue,qqp	splits: train: 363846 rows, test: 390965 rows, val: 40430 rows
dataset: glue,qqp	split: train	columns: question1, question2, label, idx
dataset: glue,qqp	labels: not_duplicate, duplicate
</pre>

```sh
# summarize the qqp subtask of glue (from PaddleHub)
gft_summary --data P:glue,qqp 2>/dev/null
```
<pre>
dataset: glue,qqp	splits: train: 363846 rows, test: 390965 rows, val: 40430 rows
dataset: glue,qqp	split: train	columns: sentence1, sentence2, labels
dataset: glue,qqp	labels: 0, 1
</pre>

```sh
# summarize a model and a dataset
model=H:pysentimiento/robertuito-emotion-analysis
gft_summary --data H:emotion --model $model 2>/dev/null
```
<pre>
model: pysentimiento/robertuito-emotion-analysis	labels: others, joy, sadness, anger, surprise, disgust, fear
dataset: emotion	split: train	columns: text, label
dataset: emotion	labels: sadness, joy, love, anger, fear, surprise
</pre>

```sh
# How do I find the good stuff?
# Return a list of models for a particular task (sorted by downloads)
gft_summary --task H:ASR --model H:__infer__ 2>/dev/null | head
```
<pre>
task: AutomaticSpeechRecognition --> 1643 models
task: AutomaticSpeechRecognition	model: facebook/wav2vec2-base-960h	downloads: 1092953	likes: 25
task: AutomaticSpeechRecognition	model: facebook/hubert-large-ls960-ft	downloads: 65234	likes: 8
task: AutomaticSpeechRecognition	model: facebook/wav2vec2-large-960h-lv60-self	downloads: 64410	likes: 9
task: AutomaticSpeechRecognition	model: facebook/wav2vec2-large-xlsr-53	downloads: 55869	likes: 12
task: AutomaticSpeechRecognition	model: hf-internal-testing/processor_with_lm	downloads: 20777	likes: 0
task: AutomaticSpeechRecognition	model: pyannote/voice-activity-detection	downloads: 18959	likes: 4
task: AutomaticSpeechRecognition	model: patrickvonplaten/wavlm-libri-clean-100h-base-plus	downloads: 14783	likes: 1
task: AutomaticSpeechRecognition	model: facebook/wav2vec2-large-960h	downloads: 13622	likes: 3
task: AutomaticSpeechRecognition	model: facebook/s2t-small-librispeech-asr	downloads: 11808	likes: 5
</pre>

```sh
# How do I find the good stuff?
# Return a list of models for a particular dataset (sorted by downloads)
gft_summary --data H:common_voice,en --model H:__infer__ 2>/dev/null | head
```
<pre>
dataset: common_voice,en	splits: train: 564337 rows, test: 16164 rows, other: 169895 rows, invalidated: 189562 rows, val: 16164 rows
dataset: common_voice,en	split: train	columns: client_id, path, audio, sentence, up_votes, down_votes, age, gender, accent, locale, segment
dataset: common_voice --> 542 models
dataset: common_voice	model: facebook/wav2vec2-large-xlsr-53	downloads: 55869	likes: 12
dataset: common_voice	model: jonatasgrosman/wav2vec2-large-xlsr-53-english	downloads: 11633	likes: 6
dataset: common_voice	model: facebook/wav2vec2-xls-r-300m	downloads: 10156	likes: 16
dataset: common_voice	model: vumichien/wav2vec2-large-xlsr-japanese-hiragana	downloads: 8454	likes: 2
dataset: common_voice	model: jonatasgrosman/wav2vec2-large-xlsr-53-german	downloads: 6338	likes: 4
dataset: common_voice	model: jonatasgrosman/wav2vec2-large-xlsr-53-russian	downloads: 4689	likes: 6
dataset: common_voice	model: facebook/wav2vec2-large-xlsr-53-german	downloads: 4498	likes: 0
</pre>

```sh
# find some models associated with a dataset (and sort by downloads)
gft_summary --data H:emotion --model H:__infer__ 2>/dev/null | head
```
<pre>
dataset: emotion	split: train	columns: text, label
dataset: emotion	labels: sadness, joy, love, anger, fear, surprise
dataset: emotion --> 69 models
dataset: emotion	model: bhadresh-savani/distilbert-base-uncased-emotion	downloads: 505632	likes: 20
dataset: emotion	model: nateraw/bert-base-uncased-emotion	downloads: 5836	likes: 1
dataset: emotion	model: mrm8488/t5-base-finetuned-emotion	downloads: 3927	likes: 3
dataset: emotion	model: mrm8488/t5-small-finetuned-emotion	downloads: 1580	likes: 0
dataset: emotion	model: bhadresh-savani/roberta-base-emotion	downloads: 1432	likes: 0
dataset: emotion	model: bhadresh-savani/bert-base-uncased-emotion	downloads: 587	likes: 1
dataset: emotion	model: lewiswatson/distilbert-base-uncased-finetuned-emotion	downloads: 217	likes: 0
</pre>

```sh
gft_summary --model H:__contains__msmarco --topn 20
```
<pre>
# found: 143 matches; truncating to 20
substring: msmarco --> 20 models
substring: msmarco	model: sentence-transformers/msmarco-distilbert-dot-v5	downloads: 60764	likes: 1	tags: pytorch, distilbert, feature-extraction, arxiv:1908.10084, sentence-transformers, sentence-similarity, transformers, infinity_compatible
substring: msmarco	model: sentence-transformers/msmarco-distilbert-base-v3	downloads: 23369	likes: 1	tags: pytorch, distilbert, feature-extraction, arxiv:1908.10084, sentence-transformers, license:apache-2.0, sentence-similarity, transformers, infinity_compatible
substring: msmarco	model: amberoad/bert-multilingual-passage-reranking-msmarco	downloads: 21541	likes: 1	tags: pytorch, tf, jax, bert, text-classification, multilingual, dataset:msmarco, arxiv:1901.04085, transformers, msmarco, passage reranking, license:apache-2.0, infinity_compatible
substring: msmarco	model: sentence-transformers/msmarco-distilbert-base-v4	downloads: 17051	likes: 1	tags: pytorch, distilbert, feature-extraction, arxiv:1908.10084, sentence-transformers, license:apache-2.0, sentence-similarity, transformers, infinity_compatible
substring: msmarco	model: sentence-transformers/distilroberta-base-msmarco-v2	downloads: 16063	likes: 0	tags: pytorch, roberta, feature-extraction, arxiv:1908.10084, sentence-transformers, license:apache-2.0, sentence-similarity, transformers, infinity_compatible
substring: msmarco	model: sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco	downloads: 10493	likes: 6	tags: pytorch, distilbert, feature-extraction, en, dataset:ms_marco, arxiv:2104.06967, transformers, dpr, dense-passage-retrieval, knowledge-distillation, infinity_compatible
substring: msmarco	model: sentence-transformers/msmarco-distilbert-base-tas-b	downloads: 7540	likes: 3	tags: pytorch, distilbert, feature-extraction, sentence-transformers, license:apache-2.0, sentence-similarity, transformers, infinity_compatible
substring: msmarco	model: sentence-transformers/msmarco-bert-base-dot-v5	downloads: 6940	likes: 0	tags: pytorch, bert, feature-extraction, arxiv:1908.10084, sentence-transformers, sentence-similarity, transformers, infinity_compatible
substring: msmarco	model: sentence-transformers/msmarco-distilroberta-base-v2	downloads: 6178	likes: 0	tags: pytorch, jax, roberta, feature-extraction, arxiv:1908.10084, sentence-transformers, license:apache-2.0, sentence-similarity, transformers, infinity_compatible
substring: msmarco	model: nboost/pt-tinybert-msmarco	downloads: 5821	likes: 0	tags: pytorch, jax, bert, transformers, infinity_compatible
substring: msmarco	model: sentence-transformers/msmarco-distilbert-base-v2	downloads: 3785	likes: 0	tags: pytorch, distilbert, feature-extraction, arxiv:1908.10084, sentence-transformers, license:apache-2.0, sentence-similarity, transformers, infinity_compatible
substring: msmarco	model: BeIR/query-gen-msmarco-t5-large-v1	downloads: 3729	likes: 4	tags: pytorch, jax, t5, text2text-generation, transformers, autonlp_compatible
substring: msmarco	model: castorini/doc2query-t5-base-msmarco	downloads: 3629	likes: 2	tags: pytorch, jax, t5, text2text-generation, transformers, autonlp_compatible
substring: msmarco	model: sentence-transformers/msmarco-distilbert-cos-v5	downloads: 3072	likes: 1	tags: pytorch, distilbert, feature-extraction, arxiv:1908.10084, sentence-transformers, sentence-similarity, transformers, infinity_compatible
substring: msmarco	model: sentence-transformers/msmarco-MiniLM-L6-cos-v5	downloads: 2678	likes: 1	tags: pytorch, jax, bert, feature-extraction, arxiv:1908.10084, sentence-transformers, sentence-similarity, transformers, infinity_compatible
substring: msmarco	model: sentence-transformers/msmarco-distilbert-base-dot-prod-v3	downloads: 2247	likes: 0	tags: pytorch, distilbert, feature-extraction, arxiv:1908.10084, sentence-transformers, license:apache-2.0, sentence-similarity, transformers, infinity_compatible
substring: msmarco	model: castorini/monot5-base-msmarco	downloads: 1868	likes: 0	tags: pytorch, jax, t5, text2text-generation, transformers, autonlp_compatible
substring: msmarco	model: sentence-transformers/msmarco-roberta-base-ance-firstp	downloads: 1766	likes: 0	tags: pytorch, roberta, feature-extraction, sentence-transformers, license:apache-2.0, sentence-similarity, infinity_compatible
substring: msmarco	model: castorini/tct_colbert-v2-msmarco-cqe	downloads: 1103	likes: 1	tags: pytorch, bert, feature-extraction, transformers, infinity_compatible
substring: msmarco	model: castorini/unicoil-msmarco-passage	downloads: 954	likes: 1	tags: pytorch, bert, feature-extraction, transformers, infinity_compatible
</pre>

```sh
# find some models associated with a task (and sort by downloads)
gft_summary --task H:classify --model H:__infer__ 2>/dev/null | head
```
<pre>
task: TextClassification --> 3542 models
task: TextClassification	model: cross-encoder/ms-marco-MiniLM-L-12-v2	downloads: 13619654	likes: 3
task: TextClassification	model: distilbert-base-uncased-finetuned-sst-2-english	downloads: 4241299	likes: 37
task: TextClassification	model: facebook/bart-large-mnli	downloads: 1022083	likes: 88
task: TextClassification	model: nlptown/bert-base-multilingual-uncased-sentiment	downloads: 693837	likes: 29
task: TextClassification	model: cardiffnlp/twitter-roberta-base-sentiment	downloads: 608351	likes: 40
task: TextClassification	model: finiteautomata/beto-sentiment-analysis	downloads: 582167	likes: 6
task: TextClassification	model: cross-encoder/nli-distilroberta-base	downloads: 581897	likes: 5
task: TextClassification	model: roberta-large-mnli	downloads: 573280	likes: 11
task: TextClassification	model: bhadresh-savani/distilbert-base-uncased-emotion	downloads: 505632	likes: 20
</pre>

```sh
# ditto, but for machine translation
gft_summary --task H:translation --model H:__infer__ 2>/dev/null | head
```
<pre>
task: Translation --> 1470 models
task: Translation	model: Helsinki-NLP/opus-mt-zh-en	downloads: 7606505	likes: 17
task: Translation	model: t5-small	downloads: 986817	likes: 5
task: Translation	model: t5-base	downloads: 812813	likes: 32
task: Translation	model: Helsinki-NLP/opus-mt-en-de	downloads: 721507	likes: 2
task: Translation	model: Helsinki-NLP/opus-mt-en-ROMANCE	downloads: 435634	likes: 1
task: Translation	model: Helsinki-NLP/opus-mt-ar-en	downloads: 185275	likes: 3
task: Translation	model: Helsinki-NLP/opus-mt-es-en	downloads: 175059	likes: 6
task: Translation	model: Helsinki-NLP/opus-mt-de-en	downloads: 144799	likes: 4
task: Translation	model: Helsinki-NLP/opus-mt-fr-en	downloads: 141266	likes: 1
</pre>

```sh
# ditto, but for speech recognition
gft_summary --task H:ctc --model H:__infer__ 2>/dev/null | head
```
<pre>
task: AutomaticSpeechRecognition --> 1644 models
task: AutomaticSpeechRecognition	model: facebook/wav2vec2-base-960h	downloads: 1014473	likes: 24
task: AutomaticSpeechRecognition	model: facebook/wav2vec2-large-xlsr-53	downloads: 71009	likes: 12
task: AutomaticSpeechRecognition	model: facebook/wav2vec2-large-960h-lv60-self	downloads: 64331	likes: 9
task: AutomaticSpeechRecognition	model: facebook/hubert-large-ls960-ft	downloads: 63556	likes: 8
task: AutomaticSpeechRecognition	model: hf-internal-testing/processor_with_lm	downloads: 20382	likes: 0
task: AutomaticSpeechRecognition	model: pyannote/voice-activity-detection	downloads: 18946	likes: 4
task: AutomaticSpeechRecognition	model: patrickvonplaten/wavlm-libri-clean-100h-base-plus	downloads: 14700	likes: 1
task: AutomaticSpeechRecognition	model: facebook/s2t-small-librispeech-asr	downloads: 13377	likes: 5
task: AutomaticSpeechRecognition	model: facebook/wav2vec2-large-960h	downloads: 12904	likes: 3
</pre>

```sh
# ditto, but for image classification
gft_summary --task H:image-classification --model H:__infer__ 2>/dev/null | head
```
<pre>
task: ImageClassification --> 278 models
task: ImageClassification	model: google/vit-base-patch16-224	downloads: 23927	likes: 27
task: ImageClassification	model: facebook/deit-base-distilled-patch16-224	downloads: 13114	likes: 2
task: ImageClassification	model: microsoft/beit-base-patch16-224-pt22k-ft22k	downloads: 12491	likes: 1
task: ImageClassification	model: microsoft/beit-base-patch16-224	downloads: 10130	likes: 1
task: ImageClassification	model: hf-internal-testing/tiny-random-vit	downloads: 6841	likes: 0
task: ImageClassification	model: google/vit-base-patch16-384	downloads: 3975	likes: 1
task: ImageClassification	model: lysandre/tiny-vit-random	downloads: 3664	likes: 0
task: ImageClassification	model: facebook/deit-small-distilled-patch16-224	downloads: 3352	likes: 0
task: ImageClassification	model: microsoft/swin-base-patch4-window7-224	downloads: 3178	likes: 0
</pre>

<table>
<tr><th>Task</th><th>Model 1</th><th>Model 2</th><th>Model 3</th></tr>
<tr><td>audio-classification</td><td>TalTechNLP/voxlingua107-epaca-tdnn</td><td>hf-internal-testing/tiny-random-unispeech-sat</td><td>hf-internal-testing/tiny-random-unispeech</td></tr>
<tr><td>audio-to-audio</td><td>JorisCos/ConvTasNet_Libri2Mix_sepclean_8k</td><td>speechbrain/sepformer-wsj02mix</td><td>JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k</td></tr>
<tr><td>AutomaticSpeechRecognition</td><td>facebook/wav2vec2-base-960h</td><td>facebook/hubert-large-ls960-ft</td><td>facebook/wav2vec2-large-xlsr-53</td></tr>
<tr><td>conversational</td><td>microsoft/DialoGPT-small</td><td>microsoft/DialoGPT-medium</td><td>facebook/blenderbot_small-90M</td></tr>
<tr><td>feature-extraction</td><td>openai/clip-vit-base-patch32</td><td>facebook/bart-base</td><td>monsoon-nlp/hindi-bert</td></tr>
<tr><td>Fill_Mask</td><td>bert-base-uncased</td><td>distilbert-base-uncased</td><td>roberta-base</td></tr>
<tr><td>ImageClassification</td><td>google/vit-base-patch16-224</td><td>microsoft/beit-base-patch16-224-pt22k-ft22k</td><td>facebook/deit-base-distilled-patch16-224</td></tr>
<tr><td>image-segmentation</td><td>facebook/detr-resnet-50-panoptic</td><td>mishig/tiny-detr-mobilenetsv3-panoptic</td><td>hf-internal-testing/tiny-random-beit-pipeline</td></tr>
<tr><td>image-to-text</td><td>kha-white/manga-ocr-base</td><td>sachin/vit2distilgpt2</td><td>gagan3012/ViTGPT2_vizwiz</td></tr>
<tr><td>object-detection</td><td>facebook/detr-resnet-50</td><td>mishig/tiny-detr-mobilenetsv3</td><td>facebook/detr-resnet-101</td></tr>
<tr><td>protein-folding</td><td>google-deepmind/alphafold-v2</td><td></td><td></td></tr>
<tr><td>QuestionAnswering</td><td>deepset/roberta-base-squad2</td><td>distilbert-base-cased-distilled-squad</td><td>bert-large-uncased-whole-word-masking-finetuned-squad</td></tr>
<tr><td>sentence-similarity</td><td>sentence-transformers/multi-qa-MiniLM-L6-cos-v1</td><td>sentence-transformers/paraphrase-MiniLM-L6-v2</td><td>sentence-transformers/all-MiniLM-L6-v2</td></tr>
<tr><td>speech-segmentation</td><td>osanseviero/hubert-sd</td><td></td><td></td></tr>
<tr><td>structured-data-classification</td><td>scikit-learn-examples/example</td><td>julien-c/wine-quality</td><td>keras-io/TF_Decision_Trees</td></tr>
<tr><td>summarization</td><td>sshleifer/distilbart-cnn-6-6</td><td>google/pegasus-xsum</td><td>facebook/bart-large-cnn</td></tr>
<tr><td>table-question-answering</td><td>google/tapas-base-finetuned-wtq</td><td>lysandre/tiny-tapas-random-wtq</td><td>lysandre/tiny-tapas-random-sqa</td></tr>
<tr><td>text2text-generation</td><td>facebook/m2m100_418M</td><td>facebook/mbart-large-50-one-to-many-mmt</td><td>google/mt5-base</td></tr>
<tr><td>TextClassification</td><td>cross-encoder/ms-marco-MiniLM-L-12-v2</td><td>distilbert-base-uncased-finetuned-sst-2-english</td><td>facebook/bart-large-mnli</td></tr>
<tr><td>text-generation</td><td>distilgpt2</td><td>gpt2</td><td>EleutherAI/gpt-neo-1.3B</td></tr>
<tr><td>text-to-image</td><td>flax-community/dalle-mini</td><td>osanseviero/dalle-mini-fork</td><td>gsurma/ai_dreamer</td></tr>
<tr><td>text-to-speech</td><td>facebook/fastspeech2-en-ljspeech</td><td>espnet/kan-bayashi_ljspeech_joint_finetune_conformer_fastspeech2_hifigan</td><td>espnet/kan-bayashi_ljspeech_vits</td></tr>
<tr><td>token-classification</td><td>xlm-roberta-large-finetuned-conll03-english</td><td>classla/bcms-bertic-ner</td><td>dslim/bert-base-NER</td></tr>
<tr><td>Translation</td><td>Helsinki-NLP/opus-mt-zh-en</td><td>t5-small</td><td>t5-base</td></tr>
<tr><td>voice-activity-detection</td><td>pyannote/segmentation</td><td>julien-c/voice-activity-detection</td><td></td></tr>
<tr><td>zero-shot-classification</td><td>facebook/bart-large-mnli</td><td>cross-encoder/nli-distilroberta-base</td><td>BaptisteDoyen/camembert-base-xnli</td></tr>
<tr><td>zero-shot-image-classification</td><td>hf-internal-testing/tiny-random-clip-zero-shot-image-classification</td><td></td><td></td></tr>
</table>



<table>
<tr><th># of Models</th><th>Task</th><th>Models</th></tr>
<tr><td>3949 </td><td> text-generation</td><td>	distilgpt2, gpt2, EleutherAI/gpt-neo-1.3B</td></tr>
<tr><td>3143  </td><td> text-classification  </td><td> cross-encoder/ms-marco-MiniLM-L-12-v2, distilbert-base-uncased-finetuned-sst-2-english, facebook/bart-large-mnli</td></tr>
<tr><td>2390  </td><td> fill-mask bert-base-uncased	 </td><td> distilbert-base-uncased, roberta-base</td></tr>
<tr><td>2253  </td><td> text2text-generation	 </td><td> facebook/m2m100_418M, facebook/mbart-large-50-one-to-many-mmt, google/mt5-base</td></tr>
<tr><td>1508  </td><td> ctc  </td><td> facebook/wav2vec2-base-960h, facebook/hubert-large-ls960-ft, facebook/wav2vec2-large-xlsr-53</td></tr>
<tr><td>1460  </td><td> translation </td><td>  Helsinki-NLP/opus-mt-zh-en, t5-small, t5-base</td></tr>
<tr><td>1340  </td><td> classify </td><td> 	xlm-roberta-large-finetuned-conll03-english, classla/bcms-bertic-ner, dslim/bert-base-NER</td></tr>
<tr><td>1042  </td><td> conversational	 </td><td> microsoft/DialoGPT-small, microsoft/DialoGPT-medium, facebook/blenderbot_small-90M</td></tr>
<tr><td>983  </td><td> classify_spans  </td><td> deepset/roberta-base-squad2, distilbert-base-cased-distilled-squad, bert-large-uncased-whole-word-masking-finetuned-squad</td></tr>
<tr><td>982  </td><td> feature-extraction  </td><td> feature-extraction, openai/clip-vit-base-patch32, facebook/bart-base, monsoon-nlp/hindi-bert</td></tr>
<tr><td>338  </td><td> sentence-similarity  </td><td>  sentence-transformers/multi-qa-MiniLM-L6-cos-v1, sentence-transformers/paraphrase-MiniLM-L6-v2, sentence-transformers/all-MiniLM-L6-v2</td></tr>
<tr><td>320  </td><td> summarization </td><td>  summarization, sshleifer/distilbart-cnn-6-6, google/pegasus-xsum, facebook/bart-large-cnn</td></tr>
<tr><td>261  </td><td> image-classification  </td><td> google/vit-base-patch16-224, microsoft/beit-base-patch16-224-pt22k-ft22k, facebook/deit-base-distilled-patch16-224</td></tr>
<tr><td>160  </td><td> text-to-speech  </td><td> text-to-speech	facebook/fastspeech2-en-ljspeech, espnet/kan-bayashi_ljspeech_joint_finetune_conformer_fastspeech2_hifigan, espnet/kan-bayashi_ljspeech_vits</td></tr>
<tr><td>68  </td><td> zero-shot-classification  </td><td> zero-shot-classification, facebook/bart-large-mnli, cross-encoder/nli-distilroberta-base, BaptisteDoyen/camembert-base-xnli</td></tr>
<tr><td>55  </td><td> audio-to-audio </td><td> 	JorisCos/ConvTasNet_Libri2Mix_sepclean_8k, speechbrain/sepformer-wsj02mix, JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k</td></tr>
<tr><td>54  </td><td> audio-classification  </td><td> TalTechNLP/voxlingua107-epaca-tdnn, hf-internal-testing/tiny-random-unispeech-sat, hf-internal-testing/tiny-random-unispeech</td></tr>
<tr><td>26  </td><td> table-question-answering  </td><td> google/tapas-base-finetuned-wtq, lysandre/tiny-tapas-random-wtq, lysandre/tiny-tapas-random-sqa</td></tr>
<tr><td>12  </td><td> object-detection  </td><td> facebook/detr-resnet-50, mishig/tiny-detr-mobilenetsv3, facebook/detr-resnet-101</td></tr>
<tr><td>11  </td><td> image-segmentation  </td><td> facebook/detr-resnet-50-panoptic, mishig/tiny-detr-mobilenetsv3-panoptic, hf-internal-testing/tiny-random-beit-pipeline</td></tr>
<tr><td>9  </td><td> text-to-image  </td><td> flax-community/dalle-mini, osanseviero/dalle-mini-fork, gsurma/ai_dreamer</td></tr>
<tr><td>4  </td><td> structured-data-classification  </td><td> , scikit-learn-examples/example, julien-c/wine-quality, keras-io/TF_Decision_Trees</td></tr>
<tr><td>4  </td><td> image-to-text  </td><td> kha-white/manga-ocr-base, sachin/vit2distilgpt2, gagan3012/ViTGPT2_vizwiz</td></tr>
<tr><td>2  </td><td> voice-activity-detection  </td><td> pyannote/segmentation, julien-c/voice-activity-detection</td></tr>
<tr><td>1  </td><td> zero-shot-image-classification  </td><td> hf-internal-testing/tiny-random-clip-zero-shot-image-classification</td></tr>
<tr><td>1  </td><td> speech-segmentation  </td><td> osanseviero/hubert-sd</td></tr>
<tr><td>1  </td><td> protein-folding  </td><td> google-deepmind/alphafold-v2</td></tr>
</table>
