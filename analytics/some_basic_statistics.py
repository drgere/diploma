from analytics.utils import data_loading, data_loading_json, get_file
from vocab_and_glove.sentencepice_build import loading_create
from vocab_and_glove.vocab import SpVocab

import numpy as np
import random


def print_dataset_statistics(dset, vocab=None):
    num_gloss = len(dset)
    dict_size = len(dset.vocab) if vocab is None else vocab.maxIndex()
    num_tokens = 0
    gloss_lens = []
    for i, item in enumerate(dset):
        gloss = item['gloss']
        tokens = gloss.split() if vocab is None else vocab.encode(gloss)
        gloss_lens.append(len(tokens))
        num_tokens += len(tokens)
    print(f'num. glosses: {num_gloss}; dict.size: {dict_size}; num.tokens: {num_tokens}')
    mn, q25, med, q75, mx, mean, std = get_statistics_summary(gloss_lens)
    print(f'gloss size: mean: {mean:.2f}, std: {std:.2f};  min {mn}, q25 {q25}, med {med}, q75 {q75}, max {mx}')


def print_dataset_gloss(dset, n=100, seed=28871):
    gloss = [ix['gloss'] for ix in dset]
    random.seed(seed)
    sample = random.sample(gloss, n)
    for g in sample:
        print(g)


def get_statistics_summary(vals):
    mn, mx = np.min(vals), np.max(vals)
    p = np.percentile(vals, [25, 50, 75])
    q25, med, q75 = p[0], p[1], p[2]
    mean = np.mean(vals)
    std = np.std(vals)
    return mn, q25, med, q75, mx, mean, std


def print_all_statistics(subdir, label=None):
    for lang in ['en', 'es', 'fr', 'it', 'ru']:
        for subset in ['train', 'dev']:
            print(f'DATASET {lang}.{subset}')
            print_dataset_statistics(data_loading(lang, subset, label=label, subdir=subdir))
            print()


def print_all_sp_statistics(subdir, label=None, vocab_size=8000, vocab_subset='train'):
    for lang in ['en', 'es', 'fr', 'it', 'ru']:
        vocab = loading_create(lang=lang, subset=vocab_subset, subdir=subdir, dict_size=vocab_size)
        vocab = SpVocab(vocab)
        for subset in ['train', 'dev']:
            print(f'DATASET {lang}.{subset}')
            print_dataset_statistics(data_loading(lang, subset, label=label, subdir=subdir), vocab)
            print()


def duplicates_analysis(lang='en', subset='train', subdir='orig', fullPrint=False):
    dset = data_loading_json(lang, subset, subdir=subdir)
    txt2ind = {}
    for i, item in enumerate(dset):
        txt = item['gloss']
        if txt in txt2ind: txt2ind[txt].append(i)
        else: txt2ind[txt] = [i]
    print(f'Dataset: {lang}.{subset}')
    nall, nuniq = len(dset), len(txt2ind)
    print(f' unique texts: {nuniq}, all texts: {nall}, difference: {nall-nuniq}')
    if fullPrint:
        txt_ind = [(txt, inds) for txt, inds in txt2ind.items()]
        txt_ind.sort(key=lambda x: len(x[1]), reverse=True)
        for txt, inds in txt_ind:
            print(inds, txt)


def organizers_duplicates_analysis(langs=['en', 'fr', 'ru'], subset=['gloss', 'sgns']):
    import pandas as pd
    for lang in langs:
        df = pd.io.json.read_json(path_or_buf=file_get(lang), orient="records")
        df["sgns"] = df.sgns.apply(tuple)
        df["char"] = df.char.apply(tuple)
        if lang in ['en', 'fr', 'ru']:
            df["electra"] = df.electra.apply(tuple)
        len_df = len(df)
        len_df_dedup = len(df.drop_duplicates(subset=subset))
        print(f'lang: {lang}, subset: [{";".join(subset)}],  len:{len_df}, len-ded:{len_df_dedup}, diff:{len_df_dedup - len_df}')


def dubls_all():
    for lang in ['en', 'es', 'fr', 'it', 'ru']:
        for subset in ['train', 'dev']:
            duplicates_analysis(lang, subset, fullPrint=False)


def organizers_all_duplicates():
    organizers_duplicates_analysis(langs=['es', 'it'], subset=['gloss', 'sgns'])
    organizers_duplicates_analysis(langs=['es', 'it'], subset=['gloss', 'char'])
    organizers_duplicates_analysis(langs=['es', 'it'], subset=['gloss', 'sgns', 'char'])
    print()
    organizers_duplicates_analysis(langs=['en', 'fr', 'ru'], subset=['gloss', 'sgns'])
    organizers_duplicates_analysis(langs=['en', 'fr', 'ru'], subset=['gloss', 'sgns', 'electra'])


if __name__ == '__main__':
    print_all_sp_statistics(subdir='dset_v1')
