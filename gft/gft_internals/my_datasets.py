# list of datasets
# HuggingFace: https://huggingface.co/datasets
# PaddleHub: https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_list.html

import gft
import numpy as np
import os,sys,pickle,time
from gft.gft_internals.gft_util import get_arg,parse_dataset_specification,parse_model_specification

# print('loading my_datasets.py', file=sys.stderr)


# convention: _hf --> HuggingFace; _pd --> PaddleHub

# from datasets import load_dataset as load_dataset_hf

# def import_paddle_features():
#     # print(__name__ + ': loading from paddle', file=sys.stderr)
#     from paddlenlp.datasets import load_dataset as load_dataset_pd
#     from paddlenlp.datasets import MapDataset as MapDataset_pd

# data should be: <keyword> : <rhs>
# where <keyword> is Custom, HuggingFace, PaddleHub
# Custom can be abbreviated as C, 
# HuggingFace can be abbreviated as H
# and PaddleHub can be abbreviated as P or Paddle.

# In all cases, there should be three splits: train, val, test
# If the names are spelled something other than train, val and test, there is a --splits arg to deal with that
# Often, val is spelled validation; there is a hack in the code below to deal with that

# Sometimes, there are only two splits, train and test.
# In that case, use the --splits arg and specify val as __select_from_train__

# args is a dict of dicts and/or classes
# see get_arg and set_arg in util.py


#  Paddle and HuggingFace have different formats for datasets
#  We need to convert the dataset to the format that agrees with the
#  provider of the model
def convert_dataset(ds, args):
    ds_provider,_ = parse_dataset_specification(args)
    model_provider,_ = parse_model_specification(args)    
    if ds_provider == model_provider: return ds
    elif model_provider == "PaddleHub": 
        # import_paddle_features()
        from paddlenlp.datasets import load_dataset as load_dataset_pd
        from paddlenlp.datasets import MapDataset as MapDataset_pd
        return {k: MapDataset_pd(v) for k, v in ds.items()}
    else: return ds

def custom_load_dataset(dataset_name, splits):
    if not dataset_name is None and dataset_name.endswith("pkl"):
        with open(fields[1], 'rb') as fd:
            return pickle.load(fd)        
    df = {}
    if splits is None: splits = 'train,val,test'
    extensions = splits.split(',')
    assert len(extensions) == 3, 'bad splits arg (expected 3 fields separated by commas): ' + splits
    for s,ext in zip(['train', 'val', 'test'], extensions):
        df[s] = dataset_name + '.' + ext
    from datasets import load_dataset as load_dataset_hf
    return load_dataset_hf('csv', data_files=df, na_filter=False)

def HuggingFace_load_dataset(dataset_name, data_dir=None): 
    from datasets import load_dataset as load_dataset_hf
    fields = dataset_name.split(',')
    if data_dir is None: return load_dataset_hf(*fields)
    else: return load_dataset_hf(*fields, data_dir=data_dir)

def PaddleHub_load_dataset_special_case(dataset_name, config, splits, info):
    print('PaddleHub_load_dataset_special_case: ' + str([dataset_name, config, splits]), file=sys.stderr)
    import paddlehub.env as hubenv

    ds = {}
    url = info['url']
    files = info['files']
    dir = info['directory']
    from paddlehub.env import DATA_HOME
    from paddlehub.utils.utils import download

    def txt2dicts(path):
        res= []
        with open(path, 'r') as fd: 
            for line in fd.read().split('\n'):
                fields = line.split()
                if len(fields) == 2:
                    filename,label = fields[0:2]
                    if 'label_list' in info:
                        label = info['label_list'][int(label)]
                    res.append({'filename': str(os.path.join(dir, filename)),
                                'label': label})
        return res          

    def extract(x, keys, i):
        res = {}
        for k in keys:
            res[k] = x[k][i]
        return res

    # The following does not work, because it assumes the values are all 1d vectors,
    # but waveforms is not
    # npz2dicts:	key: waveforms	(1600, 220500)
    # npz2dicts:	key: labels	(1600,)
    def npz2dicts(path):
        import numpy as np
        x = np.load(path)
        keys = [k for k in x]
        for k in x:
            print('npz2dicts:\tkey: ' + k + '\t' + str(x[k].shape), file=sys.stderr)
        return [extract(x, keys, i) for i in range(len(x[keys[0]]))]
  
    def my_read(path):
        if not os.path.exists(path):
            print('downloading url: ' + url, file=sys.stderr)
            t0 = time.time()
            p = os.path.dirname(path)
            os.makedirs(p, exist_ok=True)
            download(url=url, path=DATA_HOME)
            print('finished downloading url: ' + url + ' %03f seconds' % (time.time() - t0), file=sys.stderr)
            if url.endswith('tar.gz') or url.endswith('taz') or url.endswith('taz'):
                tfile = os.path.join(DATA_HOME, os.path.basename(url))
                import tarfile
                fd = tarfile.open(tfile)
                fd.extractall(DATA_HOME)
                fd.close()
                os.remove(tfile)                
        dir = os.path.dirname(path)
        if path.endswith('.txt'): return txt2dicts(path)
        elif path.endswith('.npz'): return npz2dicts(path)
        else: assert False, 'my_read, case not written yet: ' + str(path)

    for i,s in enumerate(info['splits']):
        if splits is None or s in splits:
            file = files[i]
            p = os.path.join(hubenv.DATA_HOME, dir, file)
            ds[s] = my_read(p)
    return ds                

def PaddleHub_load_dataset(dataset_name, config, splits):
    print('PaddleHub_load_dataset: ' + str([dataset_name, config, splits]), file=sys.stderr)
    # import_paddle_features()
    from paddlenlp.datasets import load_dataset as load_dataset_pd
    from gft.gft_internals.paddle_dataset_zoo import paddle_dataset_infos
    fields = dataset_name.split(',')
    infos = paddle_dataset_infos(*fields)
    assert len(infos) > 0, 'expected at least 1 result, found: ' + str(infos)
    info = infos[0]

    if 'url' in info:
        return PaddleHub_load_dataset_special_case(dataset_name, config, splits, info)

    if not config is None and not config == 'Default':
        fields = [fields[0], config]
    ds = {}
    if splits is None: splits = ','.join(info['splits'])
    for split in splits.split(','):
        if split != '__select_from_train__':
            ds[split] = load_dataset_pd(*fields, splits=split)
    return ds

def TensorFlow_load_dataset(dataset_name, splits):
    import tensorflow_datasets as tfds
    fields = dataset_name.split(',')
    ds = {}
    if splits is None: splits = 'train,dev,test'
    for split in splits.split(','):
        if split != '__select_from_train__':
            ds[split] = tfds.load(*fields, split=split, shuffle_files=True)
    return ds

def my_load_dataset(args, data_key=None, config=None, default_splits=None):

    splits = get_arg(args, 'splits', default=None)
    if splits is None:
        splits = default_splits
    print('my_load_dataset, splits ' + str(splits) + ', default_splits: ' + str(default_splits), file=sys.stderr)

    # data = get_arg(args, 'data', default=None)
    provider,dataset_name = parse_dataset_specification(args)
    print('my_load_dataset, provider: ' + str(provider), file=sys.stderr)

    if not data_key is None:
        dataset_name = data_key
    
    if provider == "Custom":
        if splits is None: splits = 'train,val,test'
        df = custom_load_dataset(dataset_name, splits)
    elif provider == 'PaddleHub':
        df = PaddleHub_load_dataset(dataset_name, config, splits)
        # df = PaddleHub_load_dataset(dataset_name, get_arg(args, 'splits', default='train,dev,test'))
    elif provider == "HuggingFace":
        df = HuggingFace_load_dataset(dataset_name, data_dir=get_arg(args, 'data_dir', default=None))
    else: assert False, "should not get here"

    splits = get_arg(args, 'splits', None)
        
    if not splits is None:
        fields = splits.split(',')
        assert len(fields) == 3, 'bad splits arg (expected 3 fields separated by commas): ' + splits
        train,val,test = fields[0:3]

        if val == '__select_from_train__':
            tt = np.array([i for i in df['train']])
            np.random.shuffle(tt)
            p = int(len(tt) // 10)
            df['train'] = tt[p:]
            df['val'] = tt[0:p]
            df['test'] = df[test]
        else:
            print('splits: ' + str(splits), file=sys.stderr)
            print('df.keys(): ' + str([i for i in df]), file=sys.stderr)
            assert train in df, 'confusion in my_datasets'
            assert val in df, 'confusion in my_datasets'
            assert test in df, 'confusion in my_datasets'

            df['train'] = df[train]
            df['val'] = df[val]
            df['test'] = df[test]

    if not 'val' in df:
        if 'validation' in df: df['val'] = df.pop('validation')
        elif 'dev' in df: df['val'] = df.pop('dev')
        else: print('Warning, cannot file validation set in dataset: ' + str(dataset_name), file=sys.stderr)

    # Trim a number of training examples
    # if args.debug:
    #     for split in df.keys():
    #         df[split] = df[split][0:100]

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    return convert_dataset(df, args)
