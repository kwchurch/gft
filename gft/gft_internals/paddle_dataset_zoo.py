import gft
import sys

# paddle_datasets (below) is based on this:
# https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/datasets

# See this link for some descriptions of some of these datasets
# https://programming.vip/docs/text-similarity-competition.html
# That link has sections on: LCQMC, BQ Corpus, PAWS-X (Chinese)


paddle_datasets = [{'name': 'flowers', 'configs': [], 'splits': ['train', 'val'], 
                    'url': 'https://bj.bcebos.com/paddlehub-dataset/flower_photos.tar.gz',
                    'files': ['train_list.txt', 'validate_list.txt'],
                    'directory' : 'flower_photos',
                    'label_list': ["roses", "tulips", "daisy", "sunflowers", "dandelion"],
                    'num_class' : 5,},

                   # The following does not work, because it assumes the values are all 1d vectors,
                   # but waveforms is not
                   # npz2dicts:	key: waveforms	(1600, 220500)
                   # npz2dicts:	key: labels	(1600,)
                   {'name': 'esc50', 'configs': [], 'splits': ['train', 'val'], 
                    'files': ['train.npz', 'dev.npz'],
                    'directory': "esc50",
                    'url' : "https://bj.bcebos.com/paddlehub-dataset/esc50.tar.gz",
                    'label_list' : [
                        # Animals
                        'Dog', 'Rooster', 'Pig', 'Cow', 'Frog', 'Cat', 'Hen', 'Insects (flying)', 'Sheep', 'Crow',
                        # Natural soundscapes & water sounds
                        'Rain', 'Sea waves', 'Crackling fire', 'Crickets', 'Chirping birds', 'Water drops', 'Wind', 'Pouring water', 'Toilet flush', 'Thunderstorm',
                        # Human, non-speech sounds
                        'Crying baby', 'Sneezing', 'Clapping', 'Breathing', 'Coughing', 'Footsteps', 'Laughing', 'Brushing teeth', 'Snoring', 'Drinking, sipping',
                        # Interior/domestic sounds
                        'Door knock', 'Mouse click', 'Keyboard typing', 'Door, wood creaks', 'Can opening', 'Washing machine', 'Vacuum cleaner', 'Clock alarm', 'Clock tick', 'Glass breaking',
                        # Exterior/urban noises
                        'Helicopter', 'Chainsaw', 'Siren', 'Car horn', 'Engine', 'Train', 'Church bells', 'Airplane', 'Fireworks', 'Hand saw',],
                    },

                   {'name': "advertisegen", 'configs': [], 'splits': ['train', 'dev'] },
                   {'name': "bq_corpus", 'configs': [], 'splits': ['train', 'dev', 'test'] },
                   {'name': "bstc", 'configs': ['transcription_translation', 'asr'], 'splits': ['train', 'dev'] },
                   {'name': "c3", 'configs': [], 'splits': ['train', 'dev', 'test'] },
                   {'name': "cail2019_scm", 'configs': [],  'splits': ['train', 'dev', 'test'] },
                   {'name': "chnsenticorp", 'configs': [], 'splits': ['train', 'dev', 'test'] },
                   {'name': "clue", 'configs': ['afqmc', 'tnews', 'iflytek', 'ocnli', 'cmnli', 'cluewsc2020', 'csl'] , 'splits': ['train', 'dev', 'test'] },
                   {'name': "cmrc2018", 'configs': [] , 'splits': ['train', 'dev', 'trial'] },
                   {'name': "cnn_dailymail", 'configs': ['cnn', 'dailymail'] , 'splits': ['train', 'dev', 'test'] },
                   {'name': "conll2002", 'configs': []  , 'splits': ['train', 'dev', 'test'] },
                   {'name': "cote", 'configs': ['dp', 'bd', 'mfw']  , 'splits': ['train', 'test'] },
                   {'name': "couplet", 'configs': []  , 'splits': ['train', 'dev', 'test'] },
                   {'name': "drcd", 'configs': []  , 'splits': ['train', 'dev', 'test'] },
                   {'name': "duconv", 'configs': []  , 'splits': ['train', 'dev', 'test_1', 'test_2'] },
                   {'name': "dureader_checklist", 'configs': []  , 'splits': ['train', 'dev', 'test1'] },
                   {'name': "dureader_qg", 'configs': []  , 'splits': ['train', 'dev'] },
                   {'name': "dureader_robust", 'configs': []  , 'splits': ['train', 'dev', 'test'] },
                   {'name': "dureader_yesno", 'configs': []  , 'splits': ['train', 'dev', 'test'] },
                   
                   # some configs have a split for unlabeled and some do not
                   {'name': "fewclue", 
                    'configs': ['bustm', 'chid', 'iflytek', 'tnews', 'eprstmt', 'ocnli', 'csldcp', 'csl']  , 
                    'splits': ['train_0', 'train_1', 'train_2', 'train_3', 'train_4', 'train_few_all', 'dev_0', 'dev_1', 'dev_2', 'dev_3', 'dev_4', 'dev_few_all', 'test', 'test_public', 'unlabeled'] },
                   {'name': "fewclue", 
                    'configs': ['cluewsc']  , 
                    'splits': ['train_0', 'train_1', 'train_2', 'train_3', 'train_4', 'train_few_all', 'dev_0', 'dev_1', 'dev_2', 'dev_3', 'dev_4', 'dev_few_all', 'test', 'test_public'] },

                   # glue,mnli has non-standard splits
                   {'name': "glue", 'configs': ['cola', 'sst-2', 'sts-b', 'qqp', 'qnli', 'rte', 'wnli', 'mrpc']  , 'splits': ['train', 'dev', 'test'] },
                   {'name': "glue", 'configs': ['mnli']  , 'splits': ['train', 'dev_matched', 'test_matched'] },
                   {'name': "glue", 'configs': ['mnli']  , 'splits': ['train', 'dev_mismatched', 'test_mismatched'] },
                   {'name': "hyp", 'configs': []  , 'splits': ['train', 'dev', 'test'] },
                   {'name': "imdb", 'configs': []  , 'splits': ['train', 'test'] },
                   {'name': "iwslt15", 'configs': []  , 'splits': ['train', 'dev', 'test'] },
                   {'name': "lcqmc", 'configs': []  , 'splits': ['train', 'dev', 'test'] },
                   {'name': "lcsts_new", 'configs': []  , 'splits': ['train', 'dev'] },
                   {'name': "msra_ner", 'configs': []  , 'splits': ['train', 'test'] },
                   {'name': "nlpcc13_evsam05_hit", 'configs': []  , 'splits': ['train', 'dev', 'test'] },
                   {'name': "nlpcc13_evsam05_thu", 'configs': []  , 'splits': ['train', 'dev', 'test'] },
                   {'name': "nlpcc14_sc", 'configs': []  , 'splits': ['train', 'test'] },
                   {'name': "paws-x", 'configs': []  , 'splits': ['train', 'dev', 'test'] },
                   {'name': "peoples_daily_ner", 'configs': []  , 'splits': ['train', 'dev', 'test'] },
                   {'name': "poetry", 'configs': []  , 'splits': ['train', 'dev', 'test'] },
                   {'name': "ptb", 'configs': []  , 'splits': ['train', 'devid', 'test'] },
                   {'name': "seabsa16", 'configs': ['phns', 'came']  , 'splits': ['train', 'test'] },
                   {'name': "sighan-cn", 'configs': []  , 'splits': ['train', 'dev'] },
                   {'name': "squad", 'configs': []  , 'splits': ['train', 'dev_v1', 'dev_v2'] },
                   {'name': "thucnews", 'configs': []  , 'splits': ['train', 'dev', 'test'] },
                   {'name': "triviaqa", 'configs': []  , 'splits': ['train', 'dev'] },
                   {'name': "wmt14ende", 'configs': []  , 'splits': ['train', 'dev', 'test'] },
                   {'name': "xnli_cn", 'configs': []  , 'splits': ['train', 'dev', 'test'] },
                   {'name': "xnli", 'configs': ['ar','bg','de','el','en','es','fr','hi','ru','sw','th','tr','ur','vi','zh']  , 'splits': ['train'] },
                   {'name': "yahoo_answer_100k", 'configs': [] , 'splits': ['train', 'valid', 'test'] },
                   ]

# def paddle_dataset_infos(dataset_name):
#     res = []
#     for info in paddle_datasets:
#         if info['name'] == dataset_name:
#             res.append(info)
#     return res

def paddle_dataset_infos(dataset_name, config=None):
    if dataset_name is None: return []
    # print('paddle_dataset_infos, dataset_name: ' + str(dataset_name) + ', config: ' + str(config), file=sys.stderr)
    if config is None and ',' in dataset_name:
        pieces = dataset_name.split(',')
        return paddle_dataset_infos(*pieces)
    def match(info):
        if info['name'] != dataset_name: 
            return False
        if config is None:
            return True
        configs = info['configs']
        if len(configs) == 0:
            return True
        if config in configs:
            return True
        return False
    return [info for info in paddle_datasets if match(info)]

