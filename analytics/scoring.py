import os
import tqdm
from nltk import word_tokenize as tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import collections
import itertools
import moverscore_v2 as mv_sc

os.environ["model_score"] = "distilbert-base-multilingual-cased"


def BLEU(pred, target, smoothing_function=SmoothingFunction().method4):
    return sentence_bleu([pred], target, smoothing_function=smoothing_function)


def score_corpus(sys_stream, ref_streams):

    if isinstance(sys_stream, str):
        sys_stream = [sys_stream]
    if isinstance(ref_streams, str):
        ref_streams = [[ref_streams]]
    fhs = [sys_stream] + ref_streams
    corpus_score = 0
    pbar = tqdm.tqdm(desc="MvSc.", disable=None, total=len(sys_stream))
    for lines in itertools.zip_longest(*fhs):
        if None in lines:
            raise EOFError("Source and reference streams have different lengths!")
        hypo, *refs = lines
        idf_dict_hyp = collections.defaultdict(lambda: 1.0)
        idf_dict_ref = collections.defaultdict(lambda: 1.0)
        corpus_score += mv_sc.word_mover_score(
            refs, [hypo], idf_dict_ref, idf_dict_hyp,
            stop_words=[], n_gram=1, remove_subwords=False,)[0]
        pbar.update()
    pbar.close()
    corpus_score /= len(sys_stream)
    return corpus_score


def BLEU_testing(g1, g2):
    g1, g2 = tokenize(g1), tokenize(g2)
    print(f'tokenized g1: {g1}')
    print(f'tokenized g2: {g2}')
    sense_bleu = BLEU(g1, g2)
    print(f'sense-bleu: {sense_bleu}')
    pass


def Mover_testing(g1, g2):
    print(f'tokenized g1: {g1}')
    print(f'tokenized g2: {g2}')
    mover_score = score_corpus([g1], [[g2]])
    print(f'mover score: {mover_score}')
    pass


if __name__ == '__main__':
    Mover_testing("to travel in a caravan ( procession ).", "A convoy or procession of travelers.")