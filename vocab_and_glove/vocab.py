from sentencepiece import SentencePieceProcessor

BOS, BOS_ix = "<seq>", 0
EOS, EOS_ix = "</seq>", 1
PAD, PAD_ix = "<pad/>", 2
UNK, UNK_ix = "<unk/>", 3

MASK = '<mask>'
CLS = '<cls>'
SEP = '<sep>'
PREDICT = '<pred>'
ADDIT_SPECIAL_TOKENS = [MASK, CLS, SEP, PREDICT]


class Vocab:

    def encode(self, text):
        raise NotImplementedError()

    def decode(self, ind):
        raise NotImplementedError()

    def max_index(self):
        pass


class SpVocab(Vocab):

    def __init__(self, spm_model):
        assert isinstance(spm_model, SentencePieceProcessor)
        self._spm_model = spm_model

    def max_index(self):
        return self._spm_model.vocab_size()

    def encode(self, text):
        return self._spm_model.encode(text, add_eos=True, add_bos=False)

    def decode(self, ind):
        return self._spm_model.decode(ind)

#
# class PlainVocab(Vocab):
#     pass
