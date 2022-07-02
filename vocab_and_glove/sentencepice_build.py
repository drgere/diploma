from pathlib import Path
import sentencepiece as spm
import tempfile
from vocab_and_glove.vocab import *
from settings import lang_resources_folder
from analytics.utils import data_loading_json

spm.set_random_generator_seed(1713)


def save_building(lang, subset, subdir, label, method, dict_size, spm_file):
    dset = data_loading_json(lang, subset, label=label, subdir=subdir)
    with tempfile.NamedTemporaryFile(mode='w+') as temp_fp:
        for gls in (itm['gloss'] for itm in dset): print(gls, file=temp_fp)
        temp_fp.seek(0)
        spm.SentencePieceTrainer.train(
            input=temp_fp.name, model_type=method, model_prefix=spm_file,
            vocab_size=dict_size, pad_id=PAD_ix, pad_piece=PAD, eos_id=EOS_ix, eos_piece=EOS,
            bos_id=BOS_ix, bos_piece=BOS, unk_id=UNK_ix, unk_piece=UNK,
        )


def loading_create(lang, subset, subdir=None, label=None, method='unigram', dict_size=8000):
    spm_folder = Path(lang_resources_folder)/'sentencepiece'
    spm_folder.mkdir(exist_ok=True)
    subdirl = f'-subdir[{subdir}]' if subdir else ''
    ll = f'-label[{label}]' if label else ''
    file_label = f'spm{subdirl}-{lang}-{subset}{ll}-{method}-{dict_size}'
    spm_file = spm_folder/file_label
    if not (spm_file.with_suffix('.model').exists() and spm_file.with_suffix('.vocab').exists()):
        save_building(lang, subset, subdir, label, method, dict_size, spm_file)
    spm_model = spm.SentencePieceProcessor(model_file=str(spm_file.with_suffix('.model')))
    setattr(spm_model, 'id', file_label)
    return spm_model


def sp(lang, subset, method, txt, dict_size=8000, subdir=None, label=None):
    _spm = loading_create(lang, subset, subdir, label, method, dict_size=dict_size)
    print(f'spm length: {len(_spm)}')
    print(PAD_ix, EOS_ix, BOS_ix, UNK_ix)
    toks = _spm.encode_as_pieces(txt)
    print(toks)
    idxs = _spm.encode(txt)
    print(idxs)
    print(idxs+[PAD_ix, PAD_ix])
    txtr = _spm.decode(idxs+[PAD_ix, PAD_ix])
    print(txtr)
    txtr = _spm.decode_pieces(toks)
    print(txtr)


if __name__ == '__main__':
    sp('en', 'train', 'unigram', 'do travelers travel if they are given financial incentives', subdir='dset_v1')