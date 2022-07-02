import argparse
import json
import re
from glob import glob
from analytics.utils import loading_json


def get_parser(
    parser=argparse.ArgumentParser(
        description="Run a embedding to embedding regression."
    ),
):
    parser.add_argument("--folder1", type=str)
    parser.add_argument("--folder2", type=str)
    parser.add_argument("--folderOut", type=str, default="~/")
    parser.add_argument("--idpart", type=int, default=2,
                        help="number of dot-separated strings at the start of filename that"
                             "uniquely indetify a file in both folders - used for file alignment")
    parser.add_argument("--format", type=str, choices=("dictV1", "lcase"),
                        help="the way the gloss predictions are formatted for final output")
    parser.add_argument("--strategy", type=str, choices=("fallback"), default="fallback",
                        help="strategy for combining two glosses for the same word.")
    return parser


def combine_glosses(gls1, gls2, strategy):
    gls1np = re.sub("[\(\[].*?[\)\]]", "", gls1).strip()
    gls1np = re.sub("\s+", " ", gls1np)
    gls1alpha = any(c.isalpha() for c in gls1np)
    gls2alpha = any(c.isalpha() for c in gls2)
    out = None
    if strategy == 'fallback':
        if not gls2alpha:
            out = None
        elif not gls1alpha:
            out = gls2
        elif len(gls1np) <= 2:
            out = gls2
    if out is not None:
        return True, gls2
    else:
        return False, gls1


def combine_files(f1, f2, args):
    jsn1, jsn2 = loading_json(f1), loading_json(f2)
    jsn2map = { itm['id']: itm['gloss'] for itm in jsn2 }
    jsn1.sort(key=lambda ji: ji["id"])
    result = []
    for itm in jsn1:
        id = itm['id']
        gls1 = itm['gloss']
        gls2 = jsn2map[id]
        is_replaced, glsC = combine_glosses(gls1, gls2, args.strategy)
        if is_replaced:
            print(f'{id}\n{gls1}\n{gls2}\n')
        result.append({'id':id, 'gloss':glsC})
    return result


def predict(args):
    for ff1 in glob(args.folder1+"*.json"):
        idpart = '.'.join(ff1.split('/')[-1].split('.')[:args.idpart])
        ff2 = glob(args.folder2+f'{idpart}*')[0]
        res = combine_files(ff1, ff2, args)
        outf = args.folderOut
        with open(outf+idpart+'.json', 'w') as out_file:
            json.dump(res, out_file, indent=2)
    pass


if __name__ == '__main__':
    args = get_parser().parse_args()
    predict(args)