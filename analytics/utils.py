import json
from analytics.dataset import DataJson
from pathlib import Path
from settings import dataset_folder


def data_loading(lang='en', subset='train', label=None, subdir=None):
    f = get_file(lang, subset, label, subdir)
    dset = DataJson(f)
    return dset


def data_loading_json(lang='en', subset='train', label=None, subdir=None):
    f = get_file(lang, subset, label, subdir)
    with open(f, 'r') as istr:
        dset = json.load(istr)
    return dset


def get_file(lang='en', subset='train', label=None, subdir=None):
    fdir = Path(dataset_folder)
    if subdir:
        fdir = fdir / subdir
    label_tag = f'.{label}' if label else ''
    fname = f'{lang}.{subset}{label_tag}.json'
    return fdir / fname


def data_loading_text(lang='en', subset='train', subdir=None):
    jdset = data_loading_json(lang, subset, subdir=subdir)
    res = [item['gloss'] for item in jdset]
    return res


def load_text_emdeddings(lang='en', subset='train', emb='electra', subdir=None, label=None):
    import numpy as np
    dset = data_loading_json(lang, subset, label, subdir)
    res = [(item['gloss'], np.array(item[emb])) for item in dset]
    return res


def loading_embeddings(lang='en', subset='train', emb='electra', subdir=None, label=None, unique=False):
    import numpy as np
    dset = data_loading_json(lang, subset, label, subdir)
    list_ = [np.array(item[emb]) for item in dset]
    res = np.stack(list_)
    if unique:
        res = np.unique(res, axis=0)
    return res


def loading_json(fpath):
    with open(fpath, 'r') as f:
        dset = json.load(f)
    return dset


def writing_json(out_file, json_items):
    with open(out_file, 'w') as of:
        json.dump(json_items, of, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    load_text_emdeddings()


