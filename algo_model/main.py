import os
import argparse
import gc
import itertools
import json
import csv
import pathlib
import random
import numpy as np
import torch
import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import shutil
import skopt
from copy import deepcopy
from datetime import date
from datasets import data_size_embedding

import algo_model.include.some_utils_for_models
from algo_model import some_settings
from algo_model.datasets import dataloader
from vocab_and_glove.embed_glove import glove_create_loading
from analytics.some_transforms import d_out


def get_parser(
    parser=argparse.ArgumentParser(
        description="Run a embedding to embedding regression."
    ),
):
    parser.add_argument(
        "--model",
        type=str,
        default="defmod-base",
        choices=(algo_model.models_core.model_utils.MODELS.keys()),
        help="set model name or model path",
    )
    parser.add_argument(
        "--rnn_arch",
        type=str, default="gru", choices=("gru", "lstm"),
        help="architecture of the base RNN network"
    )
    parser.add_argument(
        "--emb_size",
        type=int,
        default=256,
        help="size of the word embedding used in the model, a key parameter",
    )
    parser.add_argument(
        "--maxlen",
        type=int,
        default=256,
        help="maximum length (number of tokens) in the gloss - reflects in the dataset and in the model",
    )
    parser.add_argument(
        "--model_path",
        type=pathlib.Path,
        default=None,
        help="Path to the '.pt' file of the model to load, for prediction",
    )
    parser.add_argument(
        "--settings",
        type=str,
        default="defmod-base",
        choices=(some_settings.SETTINGS.keys()),
        help="set model name or model path",
    )
    parser.add_argument(
        "--do_htune",
        action="store_true",
        help="whether to perform hyperparameter tuning",
    )
    parser.add_argument(
        "--word_emb", type=str, default=None,
        help="if defined, type of pretrained word embeddings to use, options: glove"
    )
    parser.add_argument(
        "--vocab_lang", type=str, help="language describing the dataset from which the vocabulary is derived"
    )
    parser.add_argument(
        "--vocab_subset", type=str, default="train",
        help="subset (train, dev) describing the dataset from which the vocabulary is derived"
    )
    parser.add_argument(
        "--vocab_subdir", type=str, default=None,
        help="subdir (of the settings.dataset_folder) describing the dataset from which the vocabulary is derived"
    )
    parser.add_argument(
        "--vocab_type", type=str, default="sentencepiece",
        help="type of the vocabulary (sentencepiece or plain)"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=8000,
        help="type of the vocabulary (sentencepiece or plain)"
    )
    parser.add_argument(
        "--do_train", action="store_true", help="whether to train a model from scratch"
    )
    parser.add_argument(
        "--do_pred", action="store_true", help="whether to produce predictions"
    )
    parser.add_argument(
        "--train_file", type=pathlib.Path, help="path to the train file"
    )
    parser.add_argument(
        "--dev_file", type=pathlib.Path, help="path to the dev file")
    parser.add_argument(
        "--test_file", type=pathlib.Path, help="path to the test file")
    parser.add_argument(
        "--device",
        type=torch.device,
        default=torch.device("cpu"),
        help="path to the train file",
    )
    parser.add_argument(
        "--embedding",
        type=str,
        default="sgns",
        choices=("sgns", "char", "electra"),
        help="embedding architecture to use as target",
    )
    parser.add_argument(
        "--input_key",
        type=str,
        default="sgns",
        choices=("gloss", "sgns", "char", "electra", "allvec"),
        help="embedding architecture to use as target",
    )
    parser.add_argument(
        "--allvec_mode",
        type=str,
        default="adapt",
        choices=("adapt", "concat", "merge"),
        help="if input_key == 'allvec' (concat several gloss embs), select handling of input: "
             "adapt (adapt to word_emb size), concat (send concatenation to gate activation), "
             "merge/project emb vectors to word_emb size before gate activation",
    )
    parser.add_argument(
        "--output_key",
        type=str,
        default="char",
        choices=("gloss", "sgns", "char", "electra"),
        help="embedding architecture to use as target",
    )
    parser.add_argument(
        "--experiment_id",
        type=str,
        default="",
        help="Number of layers in model. Use optimization if < 1. ",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to run training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Size of the batch",
    )
    parser.add_argument("--scheudle", type=str, default='hugginface',
                        choices=('hugginface', 'plat'),
                        help="defines the learning rate scheudler")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="warmup period (total steps percentage) for hugginface scheudler")
    parser.add_argument("--learn_rate", type=float, default=0.0001,
                        help="initial learning rate for model opt.")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--in_dropout", type=float, default=0.05,
                        help="dropout on input data, for components that support it")
    parser.add_argument(
        "--n_head",
        type=int,
        default=0,
        help="Number of heads in model. Use optimization if < 1.",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=0,
        help="Number of layers in model. Use optimization if < 1. ",
    )
    parser.add_argument(
        "--summary_logdir",
        type=pathlib.Path,
        default=pathlib.Path("logs") / f"m2m-baseline",
        help="write logs for future analysis",
    )
    parser.add_argument(
        "--save_dir",
        type=pathlib.Path,
        default=pathlib.Path("models") / f"m2m-baseline",
        help="where to save model & vocab",
    )
    parser.add_argument(
        "--spm_model_path",
        type=pathlib.Path,
        default=None,
        help="use sentencepiece model, if required train and save it here",
    )
    parser.add_argument(
        "--pred_file",
        type=pathlib.Path,
        default=pathlib.Path("m2m-baseline-preds.json"),
        help="dataset for which the predictions will be generated",
    )
    parser.add_argument(
        "--pred_output",
        type=pathlib.Path,
        default=pathlib.Path("predictions.json"),
        help="where to save predictions",
    )
    parser.add_argument(
        "--pred_out_align",
        type=bool,
        default=False,
        help="weather to output gloss predictions aligned with orig. words in test file",
    )
    parser.add_argument(
        "--pred_mod",
        type=str,
        default="",
        choices=("", "lower"),
        help="for defmod, weather and how to modifiy input gloss before applying the model",
    )
    parser.add_argument(
        "--hparams",
        type=pathlib.Path,
        default=None,
        help="Path to the hyperparameters of the models",
    )
    parser.add_argument(
        "--modout",
        type=str,
        default=None,
        help="For defmod predict only, if not none, the type of transformation to apply to output glosses: "
             "dsetv1",
    )
    parser.add_argument(
        "--aggout",
        type=str,
        default="avg",
        choices=("avg", "sum", "eos"),
        help="For defmod, select the method for aggregating the output vector from the transformer."
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Random seed for torch, python, numpy.",
    )
    parser.add_argument(
        "--rndseed",
        type=int,
        default=713873,
        help="Random seed for torch, python, numpy.",
    )
    parser.add_argument(
        '--multitask', dest='multitask', action='store_true', default=False)
    return parser


def get_word_emb(word_emb, vocab_type, vocab_lang, vocab_subset,
                 vocab_subdir, vocab_size, emb_size):
    if word_emb is not None:
        if word_emb == 'glove' and vocab_type == 'sentencepiece':
            return glove_create_loading(lang=vocab_lang, subset=vocab_subset, subdir=vocab_subdir, dict_size=vocab_size,
                                        emb_size=emb_size)
        else:
            raise ValueError(f'unknown pretrained word emb: {word_emb}')


def train(
    train_file, dev_file, vocab_lang, vocab_subset, vocab_subdir, vocab_type,
    vocab_size=8000, model="DefmodBase", emb_size=256, maxlen=256, rnn_arch='gru',
    word_emb=None, input_key="sgns", output_key="char", summary_logdir=pathlib.Path("logs") / "e2e-htune",
    save_dir=pathlib.Path("models") / "e2e-mlp", device="cuda:0", epochs=100, scheudle='hugginface',
    batch_size=1024, learning_rate=1e-4, beta1=0.9, beta2=0.999, weight_decay=1e-6,
    patience=10, batch_accum=1, dropout=0.2, in_dropout=0.05, warmup_len=0.1,
    label_smoothing=0.0, n_head=4, n_layers=4, criterion=nn.MSELoss(),
    scoring=nn.MSELoss(), train_summaries={}, dev_summaries={}, get_inputs=lambda x,y: (x,),
    get_output=lambda x, y: y, get_predict=lambda p: p, multitask=False,
):
    save_dir = save_dir
    save_best_dir = save_dir / "best"
    for idx in range(1, 10000):
        idx = "{:04d}".format(idx)
        d = save_dir / idx
        if not (d / "model.pt").is_file():
            save_dir = d
            break
    save_best_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    s_file = save_best_dir / "all_scores.json"
    s_file_tmp = save_best_dir / ".all_scores_tmp.json"
    train_name = "train"
    dev_name = "dev"

    hparams = {
        "learning_rate": learning_rate,
        "beta1": beta1,
        "beta2": beta2,
        "weight_decay": weight_decay,
        "patience": patience,
        "dropout": dropout,
        "in_dropout": in_dropout,
        "scheudler": scheudle,
        "warmup_len": warmup_len,
        "label_smoothing":label_smoothing,
        "batch_accum": int(batch_accum),
        "n_head": int(n_head),
        "n_layers": int(n_layers),
        "vocab_lang": vocab_lang,
        "vocab_subset": vocab_subset,
        "vocab_subdir": vocab_subdir,
        "vocab_type": vocab_type,
        "vocab_size": vocab_size,
        "maxlen": maxlen,
        "emb_size": emb_size,
        "word_emb": word_emb,
    }

    summary_writer = SummaryWriter(summary_logdir / idx)
    train_step = itertools.count()

    gloss_vec_size, train_dataloader = dataloader('train', data_file=args.train_file, vocab_lang=vocab_lang,
                                                  vocab_subset=vocab_subset, vocab_subdir=vocab_subdir,
                                                  vocab_type=vocab_type, vocab_size=vocab_size,
                                                  input_key=args.input_key, output_key=args.output_key,
                                                  batch_size=batch_size, shuffle=True, maxlen=maxlen)
    gloss_vec_size, dev_dataloader = dataloader('dev', data_file=args.train_file, vocab_lang=vocab_lang,
                                                vocab_subset=vocab_subset, vocab_subdir=vocab_subdir,
                                                vocab_type=vocab_type, vocab_size=vocab_size, input_key=args.input_key,
                                                output_key=args.output_key, batch_size=batch_size, shuffle=False,
                                                maxlen=maxlen)
    word_emb = get_word_emb(word_emb, vocab_type=vocab_type,
                            vocab_lang=vocab_lang, vocab_subset=vocab_subset,
                            vocab_subdir=vocab_subdir, vocab_size=vocab_size, emb_size=emb_size)
    if model != 'defmod-rnn':
        model = algo_model.models_core.model_utils.get_model(
            model, vocab_size=train_dataloader.dataset.max_index_vocab(),
            word_emb=word_emb, d_emb=emb_size, maxlen=maxlen,
            n_head=n_head, n_layers=n_layers, dropout=dropout)
    else:
        if args.input_key == 'allvec':
            if args.allvec_mode == 'adapt':
                allvec = None
                d_input = gloss_vec_size
                d_allvec = -1
            else:
                allvec = args.allvec_mode
                d_input = data_size_embedding
                d_allvec = gloss_vec_size
        else:
            allvec = False
            d_allvec = -1
            d_input = data_size_embedding
        model = algo_model.models_core.model_utils.get_model(
            model, base_arch=rnn_arch, vocab_size=train_dataloader.dataset.max_index_vocab(),
            word_emb=word_emb, d_emb=emb_size, d_input=d_input, maxlen=maxlen,
            n_layers=n_layers, net_dropout=dropout, in_dropout=in_dropout, use_gateact=True,
            allvec=allvec, d_allvec=d_allvec)

    print('MODEL:')
    print(model)
    model = model.to(device)
    model.train()

    ignore_index = model.padding_idx if hasattr(model, 'padding_idx') else None

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
    )
    
    input_tensor_key = f"{input_key}_tensor"
    output_tensor_key = f"{output_key}_tensor"
    
    best_score = float("inf")
    strikes = 0

    epochs_range = tqdm.trange(epochs, desc="Epochs")
    total_steps = (len(train_dataloader) * epochs) // batch_accum
    if scheudle == 'hugginface':
        scheduler = algo_model.models_core.model_utils.sched(optimizer, round(total_steps * warmup_len), total_steps)
    elif scheudle == 'plat':
        scheduler = algo_model.models_core.model_utils.sched_plateau(optimizer)

    for epoch in epochs_range:

        pbar = tqdm.tqdm(
            desc=f"Train {epoch}", total=len(train_dataloader.dataset), disable=None, leave=False
        )
        optimizer.zero_grad()
        for i, batch in enumerate(train_dataloader):
            X = batch[input_tensor_key].to(device)
            Y = batch[output_tensor_key].to(device)
            inputs = get_inputs(X, Y)
            output = get_output(X, Y)
            if multitask:
                multiout = batch['allvec_tensor'].to(device)
                pred, multipred = model(*inputs)
                pred = get_predict(pred)
                multipred = get_predict(multipred)
            else:
                pred = get_predict( model(*inputs) )
            loss = criterion(pred, output,
                             ignore_index=ignore_index,
                             label_smoothing=label_smoothing)
            if multitask:
                loss += 0.3 * criterion(multipred, multiout,
                                ignore_index=ignore_index,
                                label_smoothing=label_smoothing)

            loss.backward()
            grad_remains = True
            step = next(train_step)
            if i % batch_accum == 0:
                optimizer.step()
                if scheudle == 'hugginface': scheduler.step()
                optimizer.zero_grad()
                grad_remains = False
                if scheudle == 'hugginface': last_lr = scheduler.get_last_lr()[0]
                elif scheudle == 'plat':
                    if hasattr(scheduler, '_last_lr'): last_lr = scheduler._last_lr[0]
                    else: last_lr = 0.0
                summary_writer.add_scalar(
                    "{}/lr".format(train_name), last_lr, step
                )
            with torch.no_grad():
                for k in train_summaries:
                    val = train_summaries[k](pred, output, ignore_index).item()
                    summary_writer.add_scalar(f"{train_name}/{k}", val, step)
                
            pbar.update(Y.size(0))
        if grad_remains:
            optimizer.step()
            if scheudle == 'hugginface': scheduler.step()
            optimizer.zero_grad()
        pbar.close()

        model.eval()
        with torch.no_grad():
            pbar = tqdm.tqdm(
                desc=f"Eval {epoch}",
                total=len(dev_dataloader.dataset),
                disable=None,
                leave=False,
            )
            sum_scores = { k:0 for k in dev_summaries }
            new_score = 0
            for batch_number, batch in enumerate(dev_dataloader):
                X = batch[input_tensor_key].to(device)
                Y = batch[output_tensor_key].to(device)
                inputs = get_inputs(X, Y)
                output = get_output(X, Y)
                if multitask:
                    pred1, multipred = model(*inputs)
                    pred = get_predict(pred1)
                else:
                    pred1 = model(*inputs)
                    pred = get_predict(pred1)

                for k in dev_summaries:
                    sum_scores[k] += dev_summaries[k](pred, output, ignore_index).item()
                new_score += scoring(pred, output, ignore_index).item()
                
                pbar.update(Y.size(0))

            dev_scores = {}
            for k in dev_summaries:
                val = sum_scores[k] / (batch_number+1)
                dev_scores[k] = val
                summary_writer.add_scalar(f"{dev_name}/{k}", val, epoch)
            new_score = new_score / (batch_number+1)

            if scheudle == 'plat': scheduler.step(new_score)

            pbar.close()
            if new_score < (best_score * 0.999):
                best_score = new_score
                model.save(save_dir / "model.pt")
                with open(save_dir / "hparams.json", "w") as json_file:
                    json.dump(hparams, json_file, indent=2)
                with open(save_dir / "best_scores.txt", "w") as score_file:
                    print(new_score, file=score_file)
                s_d = {
                    'id': idx,
                    'run_id': idx,
                    'model': model.name,
                    'date': str(date.today()),
                    'epoch': epoch,
                    'score': best_score,
                }
                s_d.update(dev_scores)
                s_d.update(hparams)
                s_data = {}
                if s_file.is_file():
                    with open(s_file, "r") as json_file:
                        s_data = json.load(json_file)
                s_data[idx] = s_d
                with open(s_file_tmp, "w") as json_file:
                    json.dump(s_data, json_file, indent=2)
                os.replace(s_file_tmp, s_file)
                strikes = 0
            else:
                strikes += 1
            if not (save_best_dir / "best_scores.txt").is_file():
                overall_best_score = float("inf")
            else:
                with open(save_best_dir / "best_scores.txt", "r") as score_file:
                    overall_best_score = float(score_file.read())
            if new_score < overall_best_score:
                model.save(save_best_dir / "model.pt")
                with open(save_best_dir / "hparams.json", "w") as json_file:
                    json.dump(hparams, json_file, indent=2)
                with open(save_best_dir / "best_scores.txt", "w") as score_file:
                    print(new_score, file=score_file)
        if strikes >= patience:
            epochs_range.close()
            break
        model.train()

    if s_file.is_file():
        with open(s_file, "r") as json_file:
            s_data = json.load(json_file)
            
        data_list = []
        for idx in s_data:
            data_list.append(s_data[idx])
        
        data_list.sort(key=lambda x:float(x['score']))
        
        for i, data in enumerate(data_list):
            idx = "{:04d}".format(i)
            shutil.move(str(save_dir.parent / (str(data['id']))),
                        str(save_dir.parent / (str(data['id']) + "_tmp")))
            shutil.move(str(summary_logdir / (str(data['id']))),
                        str(summary_logdir / (str(data['id']) + "_tmp")))
        
        score_dict_new = {}
        data_list_new = []
        for i, data in enumerate(data_list):
            idx = "{:04d}".format(i+1)
            shutil.move(str(save_dir.parent / (str(data['id']) + "_tmp")),
                       str(save_dir.parent / idx))
            shutil.move(str(summary_logdir / (str(data['id']) + "_tmp")),
                       str(summary_logdir / idx))
            data['id'] = idx
            score_dict_new[idx] = data
            data_list_new.append(data)
            
        with open(s_file.with_suffix('.csv'), 'w', encoding='UTF8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data_list_new[0].keys())
            writer.writeheader()
            writer.writerows(data_list)
            
        with open(s_file_tmp, "w") as json_file:
            json.dump(score_dict_new, json_file, indent=2)
        os.replace(s_file_tmp, s_file)

    gc.collect()
    return best_score


def pred(args):
    stts = some_settings.SETTINGS[args.settings]()
    pred_converter = stts.test_pred_handler()
    dataloader = None
    
    if args.model_path:
        if args.model_path.is_file():
            model = algo_model.models_core.model_utils.get_model(args.model_path, device=args.device)
        else:
            ValueError("Model file does not exist on given path: "
                       "{}!".format(args.model_path))  
    elif args.experiment_id:
        if pathlib.Path(args.save_dir / args.experiment_id / "model.pt"): 
            model = algo_model.models_core.model_utils.get_model(
                pathlib.Path(args.save_dir / args.experiment_id / "model.pt"), device=args.device)
        else:
            ValueError("Model file with given experiment id "
                       "does not exists: {}!".format(args.experiment_id))
    else:
        raise ValueError("There is no 'experiment_id' nor 'model_path' "
                         "arguments! Set at least one of them to address"
                         "wanted model.")
    lowercase = False
    _, dataloader = dataloader('test', data_file=args.pred_file, vocab_lang=args.vocab_lang,
                               vocab_subset=args.vocab_subset, vocab_subdir=args.vocab_subdir,
                               vocab_type=args.vocab_type, vocab_size=args.vocab_size, input_key=args.input_key,
                               output_key=args.output_key, batch_size=1, shuffle=True,
                               lowercase=lowercase)  # important! predictions are for single data points
    model.to(args.device)
    model.eval()
    input_tensor_key = f"{args.input_key}_tensor"
    predictions = []
    with torch.no_grad():
        pbar = tqdm.tqdm(desc="Pred.", total=len(dataloader.dataset))
        for batch in dataloader:
            btch = batch[input_tensor_key].to(args.device)
            pred = model.pred(btch, decode_fn=dataloader.dataset.decode)
            for id, gloss in zip(batch["id"], dataloader.dataset.decode(pred)):
                if args.modout == 'dsetv1': gloss = d_out(gloss)
                predictions.append({"id": id, args.output_key: gloss})

            pbar.update(batch[input_tensor_key].size(0))
        pbar.close()
    with open(args.pred_output, "w") as ostr: json.dump(predictions, ostr)
    if args.pred_out_align and args.output_key == "gloss":
        with open(args.pred_file, 'r') as istr: testDset = json.load(istr)
        testDset = sorted(list(testDset), key=lambda x: x['id'])
        predId2Gloss = { item['id']: item['gloss'] for item in predictions }
        outFile = args.pred_output.with_suffix('.align.txt')
        with open(outFile, 'w') as outf:
            for it in testDset:
                print(f'id: {it["id"]:13}, WORD:: {it["word"]}, gloss: {it["gloss"]}', file=outf)
                print(f'            MODEL GLOSS:: {predId2Gloss[it["id"]]}', file=outf)


def apply_rnd_seed(rndseed):
    torch.manual_seed(rndseed)
    np.random.seed(rndseed)
    random.seed(rndseed)


def main(args):
    assert not (args.do_train and args.do_htune), "Conflicting options"
    
    hparams_default={
        "learning_rate": 1e-3,
        "weight_decay": 0.5,
        "beta_a": 0.9,
        "beta_b": 0.9,
        "dropout": 0.2,
        "warmup_len": 0.2,
        "label_smoothing": 0,
        "batch_accum": 1,
        "n_head_pow": 2,
        "n_layers": 2,
    }

    if args.do_train:
        apply_rnd_seed(args.rndseed)
        # logger.debug("Performing training")
        stts = some_settings.SETTINGS[args.settings]()
        criterion = stts.cr()
        train_summaries = stts.train_sum()
        dev_summaries = stts.dev_sum()
        scoring = stts.score()
        get_inputs = stts.train_input_handler()
        get_output = stts.train_output_handler()
        get_predict = stts.train_pred_handler()

        train(
            train_file=args.train_file, dev_file=args.dev_file,
            vocab_lang=args.vocab_lang, vocab_subset=args.vocab_subset,
            vocab_subdir=args.vocab_subdir, vocab_type=args.vocab_type, vocab_size=args.vocab_size,
            model = args.model, emb_size=args.emb_size, maxlen=args.maxlen, rnn_arch=args.rnn_arch,
            word_emb=args.word_emb,
            input_key = args.input_key,
            output_key = args.output_key,
            summary_logdir = args.summary_logdir,
            save_dir = args.save_dir,
            device = args.device,
            epochs = args.epochs,
            batch_size = args.batch_size,
            patience = args.patience,
            dropout=args.dropout,
            in_dropout=args.in_dropout,
            scheudle=args.scheudle,
            learning_rate=args.learn_rate,
            warmup_len=args.warmup,
            n_head = args.n_head,
            n_layers = args.n_layers,
            criterion = criterion,
            train_summaries = train_summaries,
            dev_summaries = dev_summaries,
            scoring = scoring,
            get_inputs = get_inputs,
            get_output = get_output,
            get_predict = get_predict,
            aggout = args.aggout,
            multitask = args.multitask,
        )
    elif args.do_htune:
        stts = some_settings.SETTINGS[args.settings]()
        criterion = stts.cr()
        train_summaries = stts.train_sum()
        dev_summaries = stts.dev_sum()
        scoring = stts.score()
        search_space = stts.space()
        skopt_kwargs = stts.sk_kwargs()
        get_inputs = stts.train_input_handler()
        get_output = stts.train_output_handler()
        get_predict = stts.train_pred_handler()
        
        @skopt.utils.use_named_args(search_space)
        def train(**hparams):

            hps = deepcopy(hparams_default)
            hps.update(hparams)
            
            n_head=args.n_head if args.n_head>1 else 2**hps["n_head_pow"]
            n_layers=args.n_layers if args.n_layers>1 else hps["n_layers"]
            
            best_loss = train(
                train_file=args.train_file, dev_file=args.dev_file,
                vocab_lang=args.vocab_lang, vocab_subset=args.vocab_subset,
                vocab_subdir=args.vocab_subdir, vocab_type=args.vocab_type,
                vocab_size=args.vocab_size,
                model=args.model, emb_size=args.emb_size, maxlen=args.maxlen,
                word_emb=args.word_emb,
                input_key=args.input_key,
                output_key=args.output_key,
                summary_logdir=args.summary_logdir,
                save_dir=args.save_dir,
                device=args.device,
                epochs=args.epochs,
                batch_size = args.batch_size,
                patience = args.patience,
                dropout=hps["dropout"],
                in_dropout=args.in_dropout,
                learning_rate=hps["learning_rate"],
                beta1=min(hps["beta_a"], hps["beta_b"]),
                beta2=max(hps["beta_a"], hps["beta_b"]),
                weight_decay=hps["weight_decay"],
                batch_accum=hps["batch_accum"],
                scheudle=args.scheudle,
                warmup_len=hps["warmup_len"],
                label_smoothing=hps["label_smoothing"],
                n_head=n_head,
                n_layers=n_layers,
                criterion=criterion,
                train_summaries=train_summaries,
                dev_summaries=dev_summaries,
                scoring=scoring,
                get_inputs=get_inputs,
                get_output=get_output,
                get_predict=get_predict,
                aggout = args.aggout,
                multitask = args.multitask,
            )
            return best_loss

        result = skopt.gp_minimize(train, search_space, **skopt_kwargs)
        skopt.dump(result, args.save_dir / "results.pkl", store_objective=False)

    if args.do_pred:
        pred(args)


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
