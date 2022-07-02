import os
import tqdm
import argparse
import pathlib
import moverscore_v2 as mv_sc
from nltk.translate.bleu_score import sentence_bleu as bleu
from nltk import word_tokenize as tokenize
import out_check
import collections
import itertools
import json

os.environ["MOVERSCORE_MODEL"] = "distilbert-base-multilingual-cased"


def get_parser(parser=argparse.ArgumentParser(description="score a submission")):
    parser.add_argument(
        "submission_path",
        type=pathlib.Path,
        help="path to submission file to be scored, or to a directory of submissions to be scored",
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
    return parser


def corp_score(sys_stream, ref_streams):

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
            refs,
            [hypo],
            idf_dict_ref,
            idf_dict_hyp,
            stop_words=[],
            n_gram=1,
            remove_subwords=False,
        )[0]
        pbar.update()
    pbar.close()
    corpus_score /= len(sys_stream)
    return corpus_score


def def_eval(args, summary):
    reference_lemma_groups = collections.defaultdict(list)
    all_preds, all_tgts = [], []
    with open(args.submission_file, "r") as fp:
        submission = sorted(json.load(fp), key=lambda r: r["id"])
    with open(args.reference_file, "r") as fp:
        reference = sorted(json.load(fp), key=lambda r: r["id"])

    assert len(submission) == len(reference), "Missing items in submission!"
    id_to_lemma = {}
    pbar = tqdm.tqdm(total=len(submission), desc="S-BLEU", disable=None)
    for sub, ref in zip(submission, reference):
        assert sub["id"] == ref["id"], "Mismatch in submission and reference files!"
        all_preds.append(sub["gloss"])
        all_tgts.append(ref["gloss"])
        sub["gloss"] = tokenize(sub["gloss"])
        ref["gloss"] = tokenize(ref["gloss"])
        sub["sense-BLEU"] = bleu([sub["gloss"]], ref["gloss"])
        reference_lemma_groups[(ref["word"], ref["pos"])].append(ref["gloss"])
        id_to_lemma[sub["id"]] = (ref["word"], ref["pos"])
        pbar.update()
    pbar.close()
    for sub in tqdm.tqdm(submission, desc="L-BLEU", disable=None):
        sub["lemma-BLEU"] = max(
            bleu([sub["gloss"]], g)
            for g in reference_lemma_groups[id_to_lemma[sub["id"]]]
        )
    lemma_bleu_average = sum(s["lemma-BLEU"] for s in submission) / len(submission)
    sense_bleu_average = sum(s["sense-BLEU"] for s in submission) / len(submission)
    moverscore_average = corp_score(all_preds, [all_tgts])
    with open(args.output_file, "a") as ostr:
        print(f"MoverScore_{summary.lang}:{moverscore_average}", file=ostr)
        print(f"BLEU_lemma_{summary.lang}:{lemma_bleu_average}", file=ostr)
        print(f"BLEU_sense_{summary.lang}:{sense_bleu_average}", file=ostr)
    return (
        args.submission_file,
        moverscore_average,
        lemma_bleu_average,
        sense_bleu_average,
    )


def main(args):
    def do_score(submission_file, summary):
        args.submission_file = submission_file
        args.reference_file = (
            args.reference_files_dir
            / f"{summary.lang}.test.{summary.track}.complete.json"
        )
        eval_func = def_eval
        eval_func(args, summary)

    if args.output_file.is_dir():
        args.output_file = args.output_file / "scores.txt"
    open(args.output_file, "w").close()
    if args.submission_path.is_dir():
        files = list(args.submission_path.glob("*.json"))
        assert len(files) >= 1, "No data to score!"
        summaries = [out_check.main(f) for f in files]
        assert len(set(summaries)) == len(files), "Ensure files map to unique setups."
        rd_cfg = [
            (s.lang, a) for s in summaries if s.track == "revdict" for a in s.vec_archs
        ]
        assert len(set(rd_cfg)) == len(rd_cfg), "Ensure files map to unique setups."
        for summary, submitted_file in zip(summaries, files):
            do_score(submitted_file, summary)
    else:
        summary = out_check.main(args.submission_path)
        do_score(args.submission_path, summary)


if __name__ == "__main__":
    main(get_parser().parse_args())
