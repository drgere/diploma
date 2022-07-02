import torch
import json
from collections import defaultdict
from itertools import count
import random
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

BOS = "<seq>"
EOS = "</seq>"
PAD = "<pad/>"
UNK = "<unk/>"

sup_arch = ("sgns", "char")


class DataJson(Dataset):

    def __init__(self, file, vocab=None, freeze_vocab=False, maxlen=256):
        if vocab is None:
            self.vocab = defaultdict(count().__next__)
        else:
            self.vocab = defaultdict(count(len(vocab)).__next__)
            self.vocab.update(vocab)
        pad, eos, bos, unk = (
            self.vocab[PAD],
            self.vocab[EOS],
            self.vocab[BOS],
            self.vocab[UNK],
        )
        if freeze_vocab:
            self.vocab = dict(vocab)
        with open(file, "r") as istr:
            self.items = json.load(istr)
        for json_dict in self.items:
            if "gloss" in json_dict:
                json_dict["gloss_tensor"] = torch.tensor(
                    [bos]
                    + [
                        self.vocab[word]
                        if not freeze_vocab
                        else self.vocab.get(word, unk)
                        for word in json_dict["gloss"].split()
                    ]
                    + [eos]
                )
                if maxlen:
                    json_dict["gloss_tensor"] = json_dict["gloss_tensor"][:maxlen]
            for arch in sup_arch:
                if arch in json_dict:
                    json_dict[f"{arch}_tensor"] = torch.tensor(json_dict[arch])
            if "electra" in json_dict:
                json_dict["electra_tensor"] = torch.tensor(json_dict["electra"])
        self.has_gloss = "gloss" in self.items[0]
        self.has_vecs = sup_arch[0] in self.items[0]
        self.has_electra = "electra" in self.items[0]
        self.itos = sorted(self.vocab, key=lambda w: self.vocab[w])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def decode(self, tensor):
        with torch.no_grad():
            if tensor.dim() == 2:
                decoded = []
                for tensor_ in tensor.t():
                    decoded.append(self.decode(tensor_))
                return decoded
            else:
                return " ".join(
                    [self.itos[i.item()] for i in tensor if i != self.vocab[PAD]]
                )

    def save(self, file):
        torch.save(self, file)

    @staticmethod
    def load(file):
        return torch.load(file)


class TokenSample(Sampler):

    # def __init__(self, dataset, data_source: Optional[Sized], batch_size=200, size_fn=len, drop_last=False):
    def __init__(self, dataset, batch_size=200, size_fn=len, drop_last=False):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.size_fn = size_fn
        self._len = None
        self.drop_last = drop_last
        self.shuffle = True

    def __iter__(self):
        indices = range(len(self.dataset))
        if self.shuffle:
            indices = list(indices)
            random.shuffle(indices)
        i = 0
        selected = []
        numel = 0
        for i in indices:
            if numel + self.size_fn(self.dataset[i]) > self.batch_size:
                if selected:
                    yield selected
                selected = []
                numel = 0
            numel += self.size_fn(self.dataset[i])
            selected.append(i)
        if selected and not self.drop_last:
            yield selected

    def __len__(self):
        if self._len is None:
            self._len = (
                sum(self.size_fn(self.dataset[i]) for i in range(len(self.dataset)))
                // self.batch_size
            )
        return self._len


def get_dataloader(dataset, batch_size=200, shuffle=True):
    has_gloss = dataset.has_gloss
    has_vecs = dataset.has_vecs
    has_electra = dataset.has_electra
    pad_idx = dataset.vocab[PAD]

    def do_collate(json_dicts):
        batch = defaultdict(list)
        for jdict in json_dicts:
            for key in jdict:
                batch[key].append(jdict[key])
        if has_gloss:
            batch["gloss_tensor"] = pad_sequence(
                batch["gloss_tensor"], padding_value=pad_idx, batch_first=False
            )
        if has_vecs:
            for arch in sup_arch:
                batch[f"{arch}_tensor"] = torch.stack(batch[f"{arch}_tensor"])
        if has_electra:
            batch["electra_tensor"] = torch.stack(batch["electra_tensor"])
        return dict(batch)

    if dataset.has_gloss:
        def do_size_item(item):
            """retrieve tensor size, to batch items per elements"""
            return item["gloss_tensor"].numel()

        return DataLoader(
            dataset,
            collate_fn=do_collate,
            batch_sampler=TokenSample(
                dataset, batch_size=batch_size, size_fn=do_size_item, shuffle=shuffle
            ),
        )
    else:
        return DataLoader(
            dataset, collate_fn=do_collate, batch_size=batch_size, shuffle=shuffle
        )
