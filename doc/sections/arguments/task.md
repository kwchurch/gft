# Task

The following generates a list of tasks:

```sh
gft_summary --task H:__contains__
```

<table>
<tr><th>Task</th><th>Aliases</th><th>Description</th></tr>
<tr><td>AudioClassification</td><td>classify_audio, audio-classification</td><td>input audio and output a class label</td></tr>
<tr><td>Audio_to_Audio</td><td></td><td></td></tr>
<tr><td>AutomaticSpeechRecognition</td><td>ASR, ctc</td><td>input audio and output a transcription; also known as ASR (automatic speech recognition), STT (speech-to-text)</td></tr>
<tr><td>Conversational</td><td></td><td></td></tr>
<tr><td>FeatureExtraction</td><td>feature-extraction</td><td></td></tr>
<tr><td>Fill_Mask</td><td></td><td>input text with a single mask (<mask>) and output fillers for <mask> with scores (probabilities)</td></tr>
<tr><td>ImageClassification</td><td>classify_images, image-classification</td><td>input image and output a class label</td></tr>
<tr><td>ImageSegmentation</td><td></td><td></td></tr>
<tr><td>Image_to_Text</td><td></td><td>input image and output text; also known as OCR (optical character recognition)</td></tr>
<tr><td>ObjectDetection</td><td></td><td></td></tr>
<tr><td>QuestionAnswering</td><td>classify_spans, QA</td><td>input question and context, and output an answer (a substring of the context); also known as classify_spans, question-answering</td></tr>
<tr><td>regress</td><td></td><td>like classify, but optimizes a different loss (mean squared error, as opposed to cross-entropy)</td></tr>
<tr><td>SentenceSimilarity</td><td>sentence-similarity</td><td></td></tr>
<tr><td>StructuredDataClassification</td><td></td><td></td></tr>
<tr><td>Summarization</td><td></td><td></td></tr>
<tr><td>TableQuestionAnswering</td><td></td><td></td></tr>
<tr><td>TextClassification</td><td>classify, text-classification</td><td>input one or two texts and output a class label (includes sentiment analysis)</td></tr>
<tr><td>TextGeneration</td><td></td><td></td></tr>
<tr><td>Text_to_Image</td><td></td><td></td></tr>
<tr><td>Text_to_Speech</td><td></td><td>input text and output audio; also known as TTS (text-to-speech), speech synthesis</td></tr>
<tr><td>TokenClassification</td><td>classify_tokens</td><td>input text and output a class label for each input token; includes NER (named entity recognition) and POS (part of speech) tagging</td></tr>
<tr><td>Translation</td><td>MT</td><td>input text in one language and output a text in another language; also known as MT (machine translation)</td></tr>
<tr><td>VoiceActivityDetection</td><td></td><td>input audio and output region that contains speech; also known as VAD and SAD (speech activity detection)</td></tr>
<tr><td>Zero_ShotClassification</td><td></td><td></td></tr>
</table>