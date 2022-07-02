import numpy as np, os
from settings import glove_folder
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import tempfile, shutil
from subprocess import call
from torchtext.vocab import GloVe, _infer_shape
from scipy.spatial.distance import cosine
from vocab_and_glove.sentencepice_build import loading_create
from analytics.utils import data_loading_text

some_rnd_seed = 42
random.seed(some_rnd_seed)


def glove_create(lang, subset, subdir, spm, out_file):
    with open(out_file, 'w') as out:
        for gloss in data_loading_text(lang, subset, subdir):
            print(' '.join(spm.encode_as_pieces(gloss)), file=out)


def vect_loading(glove_out_file, spm, emb_size, rnd_seed):

    pad_id, eos_id, bos_id, unk_id = spm.pad_id(), spm.eos_id(), spm.bos_id(), spm.unk_id()
    special_ids = [pad_id, eos_id, bos_id, unk_id]
    loaded_ids = []
    with open(glove_out_file, 'rb') as f:
        _, dim = _infer_shape(f)
        max_vectors = spm.vocab_size()
        assert dim == emb_size
        itos, vectors = [], np.zeros((max_vectors, dim), dtype=np.float32)
        for line in f:
            entries = line.rstrip().split(b" ")
            word, entries = entries[0], entries[1:]
            assert dim == len(entries)
            word = word.decode('utf-8')
            word_ix = spm.piece_to_id(word)
            vectors[word_ix] = np.array([float(x) for x in entries])
            loaded_ids.append(word_ix)
    diff = set([ix for ix in range(spm.get_piece_size())]).difference(set(loaded_ids))
    loaded_pids_start = [(ix, spm.id_to_piece(ix)) for ix in loaded_ids
                         if spm.id_to_piece(ix).startswith('▁')]
    loaded_pids_all = [(ix, spm.id_to_piece(ix)) for ix in loaded_ids]
    for dix in diff:
        if dix in special_ids: continue
        dpc = spm.id_to_piece(dix)
        hits = 0
        vectors[dix] = np.zeros(dim)
        if dpc.startswith('▁'):
            for ix, pc in loaded_pids_start:
                if pc.startswith(dpc):
                    vectors[dix] += vectors[ix]
                    hits += 1
        else:
            for ix, pc in loaded_pids_all:
                if pc.endswith(dpc):
                    vectors[dix] += vectors[ix]
                    hits += 1
        if hits > 0:
            vectors[dix] /= hits

    np.random.seed(rnd_seed)
    for ix in special_ids:
        vectors[ix] = np.random.normal(loc=0.0, scale=1.0, size=dim)
    return vectors


def glove_create_loading(
        lang='en', subset='dev', subdir=None, label=None,
        spm_method='unigram', dict_size=4000, emb_size=64,
        window_size=10, num_iter=50, rnd_seed=8127
):
    spm = loading_create(lang=lang, subset=subset, subdir=subdir, label=label, method=spm_method, dict_size=dict_size)
    label_id = f'-{label}' if label else ''
    dir_id = f'-{subdir}' if subdir else ''
    resource_id = f'glove-{lang}-{subset}{dir_id}{label_id}-method-{spm_method}-'\
                  f'dict{dict_size}-emb{emb_size}-window{window_size}-iter{num_iter}'
    out_file = Path(glove_folder)/resource_id
    if not out_file.with_suffix('.txt').exists():
        tmp_dir = tempfile.mkdtemp()
        corpus_file = Path(tmp_dir)/'corpus.txt'
        glove_create(lang, subset, subdir, spm, corpus_file)
        glove_exec = Path(glove_folder)/'GloVe/build'
        build_params = f'"{corpus_file}" "{glove_exec}" "{out_file}" "{tmp_dir}" '\
                       f'{emb_size} {window_size} {num_iter} {rnd_seed}'
        module_folder = os.path.dirname(__file__)
        build_script = f'{module_folder}/build_glove_vectors.sh'
        call(f'{build_script} {build_params}', shell=True)
        shutil.rmtree(tmp_dir)
    return vect_loading(out_file.with_suffix('.txt'), spm, emb_size, rnd_seed)


def glove_test(
        lang, subset, subdir=None, label=None, spm_method='unigram',
        dict_size=8000, use_tfidf=True, sample_size=20, num_closest=10
):
    spm = loading_create(lang, subset, subdir, label, spm_method, dict_size)
    glosses = data_loading_text(lang, subset); N = len(glosses)
    id2glove = glove_create_loading(lang, subset, subdir, label, spm_method, dict_size)
    if use_tfidf:
        corpus = [' '.join(spm.encode_as_pieces(gloss)) for gloss in glosses]
        tfidf = TfidfVectorizer(sublinear_tf=True, lowercase=False,
                                tokenizer=lambda txt: txt.split(' '))
        tfidf_corpus = tfidf.fit_transform(corpus)
        gloss_vecs = []
        for ix, gloss in enumerate(glosses):
            gloss_vec = None
            for tok in spm.encode_as_pieces(gloss):
                tfidf_weight = tfidf_corpus[ix, tfidf.vocabulary_[tok]]
                tok_vec = id2glove[spm.piece_to_id(tok)]*tfidf_weight
                if gloss_vec is None: gloss_vec = tok_vec
                else: gloss_vec += tok_vec
            gloss_vecs.append(gloss_vec)
    else:
        gloss_vecs = []
        for gloss in glosses:
            vecs = np.stack([id2glove[ix] for ix in spm.encode(gloss)])
            gloss_vecs.append(vecs.mean(0))
    sample = random.sample([ix for ix in range(N)], sample_size)
    for ix in sample:
        print(f'gloss: {glosses[ix]}')
        vec = gloss_vecs[ix]
        dists = [cosine(vec, gloss_vecs[ix2]) for ix2 in range(N)]
        sorted_ix = np.argsort(dists)
        cnt = 0
        last = glosses[ix]
        for ix2 in sorted_ix:
            if glosses[ix2] != last:
                print(f'  sim.gloss: {glosses[ix2]}')
                last = glosses[ix2]; cnt += 1
            if cnt == num_closest: break


if __name__ == '__main__':
    glove_test(lang='en', subset='train', dict_size=7000, use_tfidf=True)