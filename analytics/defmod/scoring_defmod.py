import os
import tqdm
import pathlib
import argparse
import collections
import itertools
import json
import moverscore_v2 as mv_sc
from copy import deepcopy

from analytics.utils import writing_json

os.environ["MOVERSCORE_MODEL"] = "distilbert-base-multilingual-cased"

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk import word_tokenize as tokenize


def get_parser(parser=argparse.ArgumentParser(description="score a submission")):
    parser.add_argument(
        "submission_path",
        type=pathlib.Path,
        help="path to submission file to be scored, or to a directory of submissions to be scored",
    )
    parser.add_argument("--lang", type=str)
    parser.add_argument(
        "--submission_file",
        type=pathlib.Path
    )
    parser.add_argument(
        "--reference_file",
        type=pathlib.Path
    )
    parser.add_argument(
        "--reference_files_dir",
        type=pathlib.Path,
        help="directory containing all reference files",
        default=pathlib.Path("data"),
    )
    parser.add_argument(
        "--output_file",
        type=pathlib.Path,
        help="default path to print output",
        default=pathlib.Path("scores.txt"),
    )
    parser.add_argument(
        "--dset_output_file",
        type=pathlib.Path,
        help="path to print full test dataset enriched with scores and model-output glosses",
        default=pathlib.Path("output.json"),
    )
    return parser

def BLUE(pred, target, smoothing_function=SmoothingFunction().method4):
    return sentence_bleu([pred], target, smoothing_function=smoothing_function)


def scpre_corp(sys_stream, ref_streams):

    if isinstance(sys_stream, str):
        sys_stream = [sys_stream]
    if isinstance(ref_streams, str):
        ref_streams = [[ref_streams]]
    fhs = [sys_stream] + ref_streams
    corpus_score = 0
    pbar = tqdm.tqdm(desc="MvSc.", disable=None, total=len(sys_stream))
    all_scores = []
    for lines in itertools.zip_longest(*fhs):
        if None in lines:
            raise EOFError("Source and reference streams have different lengths!")
        hypo, *refs = lines
        idf_dict_hyp = collections.defaultdict(lambda: 1.0)
        idf_dict_ref = collections.defaultdict(lambda: 1.0)
        score = mv_sc.word_mover_score(
            refs,
            [hypo],
            idf_dict_ref,
            idf_dict_hyp,
            stop_words=[],
            n_gram=1,
            remove_subwords=False,
        )[0]
        pbar.update()
        corpus_score += score
        all_scores.append(score)
    pbar.close()
    corpus_score /= len(sys_stream)
    return corpus_score, all_scores


def def_evalution(args):
    reference_lemma_groups = collections.defaultdict(list)
    all_preds, all_tgts = [], []
    with open(args.submission_file, "r") as fp:
        submission = sorted(json.load(fp), key=lambda r: r["id"])
    with open(args.reference_file, "r") as fp:
        reference = sorted(json.load(fp), key=lambda r: r["id"])
    output = deepcopy(reference)
    id_to_lemma = {}
    pbar = tqdm.tqdm(total=len(submission), desc="S-BLEU", disable=None)
    ix = 0
    for sub, ref in zip(submission, reference):
        out = output[ix]
        out["out-gloss"] = deepcopy(sub["gloss"])
        all_preds.append(sub["gloss"])
        all_tgts.append(ref["gloss"])
        sub["gloss"] = tokenize(sub["gloss"])
        ref["gloss"] = tokenize(ref["gloss"])
        sub["sense-BLEU"] = BLUE(sub["gloss"], ref["gloss"])
        out["sense-BLEU"] = sub["sense-BLEU"]
        reference_lemma_groups[(ref["word"], ref["pos"])].append(ref["gloss"])
        id_to_lemma[sub["id"]] = (ref["word"], ref["pos"])
        ix += 1
        pbar.update()
    pbar.close()
    id2out = {out["id"]:out for out in output}
    for sub in tqdm.tqdm(submission, desc="L-BLEU", disable=None):
        sub["lemma-BLEU"] = max(
            BLUE(sub["gloss"], g)
            for g in reference_lemma_groups[id_to_lemma[sub["id"]]]
        )
        id = sub["id"]
        out = id2out[id]
        out["lemma-BLEU"] = sub["lemma-BLEU"]
    BLUE_average_lem = sum(s["lemma-BLEU"] for s in submission) / len(submission)
    BLUE_average_sense = sum(s["sense-BLEU"] for s in submission) / len(submission)
    mvrscr_average, all_mvrscr = scpre_corp(all_preds, [all_tgts])
    for ix, scr in enumerate(all_moverscores): output[ix]["moverscore"] = scr
    with open(args.output_file, "a") as ostr:
        print(f"MoverScore_{args.lang}:{mvrscr_average}", file=ostr)
        print(f"BLEU_lemma_{args.lang}:{BLUE_average_lem}", file=ostr)
        print(f"BLEU_sense_{args.lang}:{BLUE_average_sense}", file=ostr)
    writing_json(args.dset_output_file, output)
    return (
        args.submission_file,
        mvrscr_average,
        BLUE_average_lem,
        BLUE_average_sense,
    )


if __name__ == "__main__":
    def_evalution(get_parser().parse_args())