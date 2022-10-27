
# white space doesn't matter
# classify|regress : y1 + y2 + ... + yn ~ x1 + x2
# There can only be any number of y variables but only 1 or 2 x variables

# examples of equations:
# classify: gold ~ Word1 + Word2
# classify2: gold ~ word1 + word2
# regress: gold ~ word1 + word2
# regress: Valence + Arousal + Dominance ~ Word
# regress: label ~ sentence1 + sentence2
# classify: label ~ premise + hypothesis

# https://huggingface.co/transformers/model_doc/auto.html
# https://huggingface.co/transformers/model_doc/bert.html
# see eqn_types, for lists of what can appear before :

import gft

# eqn_type should be one of the following:
eqn_types_list = [['c', 'classify', 'text-classification', 'sentiment_analysis', 'sentiment-analysis'],  # c is syntactic sugar for classify
                  ['r', 'regress'],   # r is syntactic sugar for regress
                  ['t', 'classify_tokens', 'ner', 'pos_tagging', 'token-classification'], # intended for NER (and parts of speech); tasks that label each token

                  # not yet implemented
                  # stub to flesh out later

                  ['classify_sentences'],
                  ['classify_spans', 'question-answering', 'QA'], # intended for SQuAD

                  ['ctc', 'ASR', 'automatic-speech-recognition'],      # intended for speech recognition (ASR)
                  ['classify_images', 'image-classification'],
                  ['classify_audio'],

                  # NOT YET IMPLEMENTED
                  ['regress_tokens'], # like classify_tokens, but over REALs, as opposed to enums (predict a REAL for each token)
                  ['regress_spans'],  # like classify_spans, but over REALs
                  ['maskedLM'],
                  ['causalLM'],
                  ['seq2seqLM'],
                  ['choose'],
                  ['seq2seq'],

                  # these will be DEPRECATED soonish
                  ['classify_classic'],  # for debugging purposes, should be the same as classify
                  ['2', 'classify2'], # 2 is syntactic sugar for classify2 (DEPRECATED)
]

taskflow_tasks = ['knowledge_mining', 'ner', 'poetry_generation',
                  'question_answering', 'lexical_analysis', 'word_segmentation',
                  'pos_tagging', 'sentiment_analysis', 'dependency_parsing',
                  'text_correction', 'text_similarity', 'dialogue']

tasks_to_eqn_types = [["audio-classification", "classify_audio"],
                      ["ASR", "ctc"],
                      ["automatic-speech-recognition", "ctc"],
                      ["conversational", None],
                      ["feature-extraction", None],
                      ["fill-mask", None],
                      ["image-classification", "classify_images"],
                      ["QA", "classify_spans"],
                      ["question-answering", "classify_spans"],
                      ["table-question-answering", None],
                      ["text2text-generation", None],
                      ["text-classification", "classify"],
                      ["sentiment-analysis", "classify"],
                      ["text-generation", None],
                      ["token-classification", "classify_tokens"],
                      ["ner", "classify_tokens"],
                      ["MT", None],
                      ["translation", None],
                      ["translation_xx_to_yy", None],
                      ["summarization", None],
                      ["zero-shot-classification", None],
                      
                      # taskflow tasks
                      ['knowledge_mining', None],
                      ['ner', "classify_tokens"],
                      ['poetry_generation', None],
                      ["question_answering", "classify_spans"],
                      ['lexical_analysis', None],
                      ['word_segmentation', None],
                      ['pos_tagging', "classify_tokens"],
                      ['sentiment_analysis', "classify"],
                      ['dependency_parsing', None],
                      ['text_correction',  None],
                      ['text_similarity',  None],
                      ['dialogue', None]]

eqn_types_to_tasks = [["classify_audio", "audio-classification"],
                      ["ctc", "automatic-speech-recognition"],
                      ["classify_images", "image-classification"],
                      ["classify_spans", "question-answering"],
                      ["classify", "text-classification"],
                      ["classify_tokens", "token-classification"]]

def infer_task_from_eqn(args):
    from gft.gft_internals.gft_util import get_arg
    eqn = parse_eqn(get_arg(args, 'eqn', default=None))
    if eqn is None: return None
    for eqn_type,task in eqn_types_to_tasks:
        if eqn_types[eqn_type] == eqn['eqn_type']:
            return task
    return None

# Auto Models:    (from https://huggingface.co/transformers/model_doc/auto.html)
# AutoModelForCausalLM,
# AutoModelForMaskedLM,
# AutoModelForSeq2SeqLM,
# AutoModelForSequenceClassification,
# AutoModelForMultipleChoice,
# AutoModelForNextSentencePrediction,
# AutoModelForTokenClassification,
# AutoModelForQuestionAnswering,
# AutoModelForImageClassification,
# AutoModelForAudioClassification,

# AutoModelForCTC,
# AutoModelForSpeechSeq2Seq,
# AutoModelForObjectDetection,
# AutoModelForImageSegmentation,

# AutoModelForCausalLM,
# AutoModelForMaskedLM,
# AutoModelForSeq2SeqLM,

# it is probably not necessary to introduce a special case for binary classification

# classify2 			AutoModelForSequenceClassification	binary classification when labels are 0 or 1
# classify  			AutoModelForSequenceClassification	multi-class classification where labels can be arbitrary strings
# regress 	 		AutoModelForSequenceClassification	labels can be a real or a vector of reals

# classify_tokens		AutoModelForTokenClassification		NER, part of speech tagging
# classify_sentences  		AutoModelForNextSentencePrediction
# classify_spans		AutoModelForQuestionAnswering		SQuAD
# regress_spans			AutoModelForQuestionAnswering		NOT YET IMPLEMENTED
# classify_images		AutoModelForImageClassification
# classify_audio		AutoModelForAudioClassification

# maskedLM			AutoModelForMaskedLM
# causalLM			AutoModelForCausalLM
# seq2seqLM			AutoModelForSeq2SeqLM
# choose			AutoModelForMultipleChoice
# seq2seq			AutoModelForCTC

eqn_types = {}
for i,ts in enumerate(eqn_types_list):
    for t in ts:
        eqn_types[t] = i

def parse_eqn(eqn):
    if eqn is None: return eqn
    for c in ' \t\n':
        eqn = eqn.replace(c, '')
    fields = eqn.split('~')
    assert len(fields) == 2, 'cannot parse: ' + eqn
    lhs,rhs = fields[0:2]
    
    lhs_fields = lhs.split(':')
    assert len(lhs_fields) == 2, 'cannot parse: ' + eqn
    
    eqn_type,y_field_names = lhs_fields[0:2]
    y_field_names = y_field_names.split('+')
    assert eqn_type in eqn_types, 'bad eqn_type: ' + eqn_type
    x_field_names = rhs.split('+')
    assert len(x_field_names) in [1,2], 'cannot parse: ' + eqn
    return {'eqn_type' : eqn_types[eqn_type],
            'y_field_names' : y_field_names,
            'x_field_names' : x_field_names}


def verify_dataset_column(datasets, col):
    for k in datasets:
        ds = datasets[k]
        if len(ds) > 0:
            # if not col in ds[0]:
                # import pdb
                # pdb.set_trace()
            assert col in ds[0], 'verify_dataset_column failed for col: ' + str(col) + ' and ds[0]: ' + str(ds[0])

def verify_dataset_columns(datasets, eqn):
    for col in eqn['x_field_names']:
        verify_dataset_column(datasets, col)
    for col in eqn['y_field_names']:
        verify_dataset_column(datasets, col)

        
