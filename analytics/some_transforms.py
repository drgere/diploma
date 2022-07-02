from copy import deepcopy
import re

from analytics.utils import *
from analytics.some_basic_statistics import print_dataset_gloss
from settings import dataset_folder


def gloss_split(gls):
    subgls = [sg.strip() for sg in gls.split(';')]
    subgls = [sg for sg in subgls if sg != '']
    return subgls


def punct_remove(gls):
    gls = gls.strip()
    if gls.endswith('.'):
        gls = gls[:-1]
    if gls.startswith(':'):
        gls = gls[1:]
    return gls.strip()


def label_remove(gls):
    tokens = gls.split()
    labels_exist = tokens[0] == '('
    if not labels_exist:
        return [], gls
    inside_lbl = False
    label = None
    labels = []
    def_start = None
    for i, tok in enumerate(tokens):
        if not inside_lbl:
            if tok == '(':
                inside_lbl = True
            elif tok == ')':
                raise Exception(f'misformed label part: {gls}')
            elif tok == ',':
                continue
            else:
                def_start = i
                break
        else:
            if tok == ')':
                if label is None: raise Exception(f'parenteses closed without label inside: {gls}')
                labels.append(label)
                label = None
                inside_lbl = False
            elif tok == '(':
                raise Exception(f'multiple consequent open paranteses: {gls}')
            else:
                if label is not None:
                    label = ' '.join([label, tok])
                else: label = tok
    labels = [l.lower() for l in labels]
    stripped_gls = ' '.join(tokens[def_start:])
    return labels, stripped_gls


def label_remove_es(gls):
    labels_exist = '|' in gls
    if not labels_exist:
        return [], gls
    if gls.count('|') > 1:
        return [], gls
    parts = gls.split('|')
    labels, gls_fin = parts[0], parts[1]
    if len(labels) > len(gls_fin)*3:
        if gls_fin.strip() != '':
            return [], gls
    gls_fin = gls_fin.strip()
    if labels == '':
        return [], gls_fin
    if labels[-1] == '.':
        labels = labels[:-1]
    labels = labels.lower()
    splitter = re.compile(r"[,\.\(\)]| y | e ")
    labels = [l.strip() for l in splitter.split(labels)]

    def prefix(s, prfx):
        for p in prfx:
            if s.startswith(p): return s[len(p):]
        return s

    labels = [prefix(l, ['en la ', 'en ']) for l in labels]
    labels = [l for l in labels if l != '']
    return labels, gls_fin


def label_remove_ru(gls):
    tokens = gls.split()
    if len(tokens) == 0 or len(tokens) == 1:
        return [], gls
    labels_exist = tokens[0].endswith('.') or tokens[1] == '.'
    if not labels_exist: return [], gls
    labels = []; start_gls_idx = None
    for i, tok in enumerate(tokens):
        if tok in ['.', ',']: continue
        elif tok.endswith('.'): labels.append(tok[:-1])
        else:
            if i < len(tokens)-1 and tokens[i+1] == '.':
                labels.append(tok)
            else:
                start_gls_idx = i
                break
    if start_gls_idx is None:
        start_gls_idx = len(tokens)
    labels = [l.lower().strip() for l in labels if l.strip() != '']
    gloss = ' '.join(tokens[start_gls_idx:])
    gloss = gloss.strip()
    return labels, gloss


def remove_labels(gls, lang):
    if lang == 'en':
        return [], gls
    elif lang == 'fr' or lang == 'it':
        return label_remove(gls)
    elif lang == 'es':
        return label_remove_es(gls)
    elif lang == 'ru':
        return label_remove_ru(gls)
    else:
        raise ValueError(f'language not supported: {lang}')


def labels_test():
    print(label_remove('this is gloss txt . '))
    print(label_remove(' ( label1 ) this is gloss txt . '))
    print(label_remove(' ( label1 ) ( label2 ) this is gloss txt . '))
    print(label_remove(' ( label1 ) , ( label2 ) this is gloss txt . '))


def labels_test_data(lang, subset, print_every=200):
    import random
    dset = data_loading_text(lang, subset); N = len(dset)
    labels = []; num_lab = 0; num_err = 0; num_empty = 0
    for gls in dset:
        try:
            labs, gls_nolab = remove_labels(gls, lang)
            if gls_nolab.strip() == '': num_empty += 1
        except Exception as e:
            num_err += 1
            print(f'error: {e} \n')
            labs, gls_nolab = [], gls
        labels.extend(labs)
        if len(labs) > 0: num_lab += 1
        if random.random() < 1.0/print_every:
            print(f'gloss: {gls}')
            print(f'labels: {labs}')
            print(f'gloss no labels: {gls_nolab}')
            print()
    print('all labels:')
    for l in sorted(set(labels)): print(l)
    print()
    print(f'num glosses {N}, num labeled glosses {num_lab}, {num_lab/N*100:.3}%')
    print(f'num errrors: {num_err}, {num_err/N*100:.3}%')
    print(f'num empty gosses: {num_empty}, {num_empty / N * 100:.3}%')

def lowercase(gls):
    return gls.lower()


def gloss_split(items):
    result = []
    for itm in items:
        orig_id = itm['id']
        splits = gloss_split(itm['gloss']); num_splits = len(splits)
        for ix, subgls in enumerate(splits):
            subitm = deepcopy(itm)
            subitm['gloss'] = subgls
            subitm['id'] = f'{orig_id}.{ix+1}' if num_splits > 1 else orig_id
            result.append(subitm)
    return result


def labels(items, lang):
    num_err = 0
    for itm in items:
        gloss = itm['gloss']
        try:
            labels, new_gls = remove_labels(gloss, lang)
        except Exception as e:
            num_err += 1
            print(f'error: {e} \n')
            labels, new_gls = [], gloss
        if new_gls == '':
            print(f'empty gloss after lab.extract, orig gloss: "{gloss}"')
        labels = ']['.join(labels)
        itm['gloss'] = new_gls
        assert 'labels' not in itm
        itm['labels'] = labels
    print(f'num. extraction errors: {num_err}')
    return items

def gloss_split_test():
    print(gloss_split('no semicolon here'))
    print(gloss_split('one gloss ; two gloss ; three .'))

def gloss_remove(items):
    new_items = []; num_empty = 0
    for itm in items:
        if itm['gloss'].strip() != '': new_items.append(itm)
        else: num_empty += 1
    print(f'num. empty glosses: {num_empty}\n')
    return new_items

def gloss_normalization(items, punct=True, lower=True):
    for ix, itm in enumerate(items):
        gls = itm['gloss']
        if punct: gls = punct_remove(gls)
        if lower: gls = lowercase(gls)
        itm['gloss'] = gls
    return items

def data_transform(lang, subset, label, sub_folder=None, punct=True, lower=True, split=True, labels=True):

    json_items = data_loading_json(lang, subset, subdir='orig')
    save_folder = Path(dataset_folder)
    if sub_folder:
        save_folder = save_folder / sub_folder
        save_folder.mkdir(exist_ok=True)
    json_items = gloss_remove(json_items)
    if labels: json_items = labels(json_items, lang)
    json_items = gloss_remove(json_items)
    if split: json_items = gloss_split(json_items)
    json_items = gloss_normalization(json_items, punct, lower)
    out_file = f'{lang}.{subset}.{label}.json' if label else f'{lang}.{subset}.json'
    out_file = save_folder / out_file
    with open(out_file, 'w') as of:
        json.dump(json_items, of, ensure_ascii=False, indent=2)

def data_reform(lang, subset, label, sub_folder=None):

    json_items = data_loading_json(lang, subset)
    save_folder = Path(dataset_folder)
    if sub_folder:
        save_folder = save_folder / sub_folder
        save_folder.mkdir(exist_ok=True)
    out_file = f'{lang}.{subset}.{label}.json' if label else f'{lang}.{subset}.json'
    out_file = save_folder / out_file
    with open(out_file, 'w') as of:
        json.dump(json_items, of, ensure_ascii=False, indent=2)

def test_transform(lang, subset):
    data_transform(lang, subset, 'slp', 'test')
    dset = data_loading_json(lang, subset, 'slp', 'test')
    print_dataset_gloss(dset, 100)

def all_reform():
    for lang in ['en', 'fr', 'it', 'es', 'ru']:
        for subset in ['dev', 'train']:
            data_reform(lang, subset, label=None, sub_folder='orig_reformatted')

def d_out(gloss):

    gloss = gloss.strip()
    tokens = gloss.split()
    if len(tokens) <= 1: return gloss
    gloss = gloss[0].upper() + gloss[1:] + ' .'
    return gloss

def create_d():
    for lang in ['en', 'fr', 'it', 'es', 'ru']:
        for subset in ['dev', 'train']:
            print(f'--------------- TRANSFORMING {lang}{subset} ---------------\n')
            data_transform(lang, subset, '', 'dset_v1')
            dset = data_loading_json(lang, subset, '', 'dset_v1')
            print_dataset_gloss(dset, 100)
            print(f'--------------- TRANSFORMING {lang}{subset} FINISHED ---------------\n\n')

def create_case():
    for lang in ['en', 'fr', 'it', 'es', 'ru']:
        for subset in ['dev', 'train']:
            print(f'--------------- TRANSFORMING {lang}{subset} ---------------\n')
            data_transform(lang, subset, label=None, sub_folder='orig_lc', punct=False, lower=True, split=False,
                           labels=False)
            dset = data_loading_json(lang, subset, '', 'orig_lc')
            print_dataset_gloss(dset, 100)
            print(f'--------------- TRANSFORMING {lang}{subset} FINISHED ---------------\n\n')

if __name__ == '__main__':
    create_case()