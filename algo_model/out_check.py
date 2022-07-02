import json
import argparse
import pathlib


def get_parser(
    parser=argparse.ArgumentParser(
        description="Verify the output format of a submission"
    ),
):
    parser.add_argument("submission_file", type=pathlib.Path, help="file to check")
    return parser


def main(filename):
    try:
        with open(filename, "r") as istr:
            items = json.load(istr)
    except ValueError:
        raise ValueError(f'File "{filename}": could not open, submission will fail.')
    else:
        for item in items:
            if "id" not in item:
                raise ValueError(
                    f'File "{filename}": one or more items do not contain an id, submission will fail.'
                )
        ids = sorted([item["id"] for item in items])
        ids = [i.split(".") for i in ids]
        langs = {i[0] for i in ids}
        if len(langs) != 1:
            raise ValueError(
                f'File "{filename}": ids do not identify a unique language, submission will fail.'
            )
        tracks = {i[-2] for i in ids}
        if len(tracks) != 1:
            raise ValueError(
                f'File "{filename}": ids do not identify a unique track, submission will fail.'
            )
        track = next(iter(tracks))
        if track != "defmod":
            raise ValueError(
                f'File "{filename}": unknown track identified {track}, submission will fail.'
            )
        lang = next(iter(langs))
        if lang not in ("en", "es", "fr", "it", "ru"):
            raise ValueError(
                f'File "{filename}": unknown language {lang}, submission will fail.'
            )
        serials = list(sorted({int(i[-1]) for i in ids}))
        if serials != list(range(1, len(ids) + 1)):
            raise ValueError(
                f'File "{filename}": ids do not identify all items in dataset, submission will fail.'
            )
        if track == "defmod" and any("gloss" not in i for i in items):
            raise ValueError(
                f'File "{filename}": some items do not contain a gloss, defmod submission will fail.'
            )


if __name__ == "__main__":
    main(get_parser().parse_args().submission_file)
