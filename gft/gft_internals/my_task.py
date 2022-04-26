import gft
import sys
from gft.gft_internals.gft_util import parse_task_specification,parse_model_specification,get_arg

# The terminology for tasks is a bit non-standard
# We allow for a number of aliases

canonicalize_tasks = [["classify_audio", "audio-classification"],
                      ["ASR", "automatic-speech-recognition"],
                      ["ctc", "automatic-speech-recognition"],
                      ["classify_images", "image-classification"],
                      ["QA", "question-answering"],
                      ["question_answering", "question-answering"],
                      ["classify_spans", "question-answering"],
                      ["classify", "text-classification"],
                      ["classify_tokens", "token-classification"],
                      ["MT", "translation"]]


canconicalized_pipeline_tags = [["AudioClassification"],
                                ["Audio_to_Audio"],
                                ["AutomaticSpeechRecognition", "ctc", "ASR", "automatic-speech-recognition"],
                                ["Conversational"],
                                ["FeatureExtraction"],
                                ["Fill_Mask", "fill-mask"],
                                ["ImageClassification", "image-classification"],
                                ["ImageSegmentation"],
                                ["Image_to_Text"],
                                ["ObjectDetection"],
                                ["QuestionAnswering", "QA", "question_answering", "question-answering", "classify_spans"],
                                ["SentenceSimilarity"],
                                ["StructuredDataClassification"],
                                ["Summarization"],
                                ["TableQuestionAnswering"],
                                ["TextClassification", "text-classification", "classify", "text_classification"],
                                ["TextGeneration"],
                                ["Text_to_Image"],
                                ["Text_to_Speech"],
                                ["TokenClassification", "ner", "classify_tokens"],
                                ["Translation", "translation", "MT"],
                                ["VoiceActivityDetection"],
                                ["Zero_ShotClassification"]]

task_help = {'AudioClassification': 'input audio and output a class label',
             'Audio_to_Audio': 'documentation to be written',
             'AutomaticSpeechRecognition': 'input audio and output a transcription; also known as ASR (automatic speech recognition), STT (speech-to-text)',
             'Conversational': 'documentation to be written',
             'FeatureExtraction': 'documentation to be written',
             'Fill_Mask': 'input text with a single mask (<mask>) and output fillers for <mask> with scores (probabilities)',
             'ImageClassification': 'input image and output a class label',
             'ImageSegmentation': 'documentation to be written',
             'Image_to_Text': 'input image and output text; also known as OCR (optical character recognition)',
             'ObjectDetection': 'documentation to be written',
             'QuestionAnswering': 'input question and context, and output an answer (a substring of the context); also known as classify_spans, question-answering',
             'SentenceSimilarity': 'documentation to be written',
             'StructuredDataClassification': 'documentation to be written',
             'Summarization': 'documentation to be written',
             'TableQuestionAnswering': 'documentation to be written',
             'TextClassification': 'input one or two texts and output a class label (includes sentiment analysis)',
             'TextGeneration': 'documentation to be written',
             'Text_to_Image': 'documentation to be written',
             'Text_to_Speech': 'input text and output audio; also known as TTS (text-to-speech), speech synthesis',
             'TokenClassification': 'input text and output a class label for each input token; includes NER (named entity recognition) and POS (part of speech) tagging',
             'Translation': 'input text in one language and output a text in another language; also known as MT (machine translation)',
             'VoiceActivityDetection': 'input audio and output region that contains speech; also known as VAD and SAD (speech activity detection)',
             'Zero_ShotClassification': 'documentation to be written'}


def summarize_task_internal(s, tt):
    for t in tt:
        if s in t.lower() and tt[0] in task_help:
            print('\t'.join([t, task_help[tt[0]]]))
            return

def summarize_task(args):
    provider,task = parse_task_specification(args)

    if task is None: return

    if '__contains__' in task:
        s = task[len('__contains__'):].lower()
        for tt in canconicalized_pipeline_tags:
            summarize_task_internal(s, tt)
    else:
        ctask = canonicalize_task(task)
        if ctask in task_help:
            print(ctask + '\t' + task_help[ctask])

def canonicalize_task(task):

    if task is None: return

    if '__contains__' in task:
        return None
    for aliases in canconicalized_pipeline_tags:
        if task in aliases:
            print('canonicalize_task: %s --> %s' % (str(task), str(aliases[0])), file=sys.stderr) 
            return aliases[0]
    print('canonicalize_task: %s --> %s' % (str(task), str(task)), file=sys.stderr)
    return task

def infer_task_from_model(args):
    from huggingface_hub import model_info
    provider,model_key = parse_model_specification(args)
    try:
        mi = model_info(model_key)
        # print('model_key: ' + str(model_key), file=sys.stderr)
        print('model_info: ' + str(mi), file=sys.stderr)
        if hasattr(mi, 'pipeline_tag'):
            return mi.pipeline_tag
    except:
        return None
    return None

# horrible hack to guess task
def infer_task(args):
    # if task is specified, use that
    provider,task = parse_task_specification(args)

    # if task can be inferred from model, use that
    if task is None:
        task = infer_task_from_model(args)

    # if task can be inferred from eqn, use that
    if task is None:
        from gft.gft_internals.parse_eqn import infer_task_from_eqn
        task = infer_task_from_eqn(args)

    # canonicalize the task
    for alias,ctask in canonicalize_tasks:
        if task == alias:
            return ctask

    return task


    
