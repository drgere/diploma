import main
import out_check
import score


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="demo script for participants")
    subparsers = parser.add_subparsers(dest="command", required=True)
    parser_main = main.get_parser(
        parser=subparsers.add_parser(
            "main", help="Run a training or prediction."
        )
    )
    parser_defmod = main.get_parser(
        parser=subparsers.add_parser(
            "defmod", help="Run a definition modeling baseline"
        )
    )
    parser_revdict = main.get_parser(
        parser=subparsers.add_parser(
            "defmod", help="Run a reverse dictionary baseline"
        )
    )
    parser_embed2embed = main.get_parser(
        parser=subparsers.add_parser(
            "embed2embed", help="Run a embed2embed (embedding to embedding) task"
        )
    )
    parser_check_output = out_check.get_parser(
        parser=subparsers.add_parser(
            "check-format", help="Check the format of a submission file"
        )
    )
    parser_score = score.get_parser(
        parser=subparsers.add_parser("score", help="evaluate a submission")
    )
    args = parser.parse_args()
    if args.command == 'defmod':
        assert args.model.startswith(args.command), "Model is not for this task." 
        assert args.settings.startswith(args.model), "Settings is not for this model." 
        main.main(args)
    elif args.command == "main":
        assert args.settings.startswith(args.model), "Settings is not for this model."        
        main.main(args)
    elif args.command == "check-format":
        out_check.main(args.submission_file)
    elif args.command == "score":
        score.main(args)
    else:
        raise ValueError('Unknown command: {} - Only options are: {}'.format(
            args.command, tasks + ["check-format", "score"]))
