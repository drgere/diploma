import json, torch, random
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict

from vocab_and_glove.sentencepice_build import loading_create
from vocab_and_glove.vocab import SpVocab, PAD_ix

sup_archs = ("sgns", "char")
data_size_embedding = 256


class data_json(Dataset):

    def __init__(self, file, vocab, maxlen=256, lowercase=False):
        self._vocab = vocab
        with open(file, "r") as istr:
            self.items = json.load(istr)
        for json_dict in self.items:
            if "gloss" in json_dict:
                if lowercase: json_dict['gloss'] = json_dict['gloss'].lower()
                json_dict['gloss_tensor'] = torch.tensor(self._vocab.encode(json_dict['gloss']))
                if maxlen: json_dict["gloss_tensor"] = json_dict["gloss_tensor"][:maxlen]
            for arch in sup_archs:
                if arch in json_dict:
                    json_dict[f"{arch}_tensor"] = torch.tensor(json_dict[arch])
            if "electra" in json_dict:
                json_dict["electra_tensor"] = torch.tensor(json_dict["electra"])
        self.has_gloss = "gloss" in self.items[0]
        self.has_vecs = sup_archs[0] in self.items[0]
        self.has_electra = "electra" in self.items[0]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def max_index_vocab(self):
        return self._vocab.maxIndex()

    @torch.no_grad()
    def decode(self, tensor):
        if tensor.dim() == 2:
            decoded = []
            for tensor_ in tensor.t():
                decoded.append(self.decode(tensor_))
            return decoded
        else:
            return self._vocab.decode(tensor.tolist())

    def save(self, file):
        torch.save(self, file)

    @staticmethod
    def load(file):
        return torch.load(file)


class token(Sampler):

    def __init__(
            self, dataset, batch_size=150, size_fn=len, drop_last=False, shuffle=True
    ):
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
        longest_len = 0
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
            self._len = round(
                sum(self.size_fn(self.dataset[i]) for i in range(len(self.dataset)))
                / self.batch_size
            )
        return self._len


def data_loader(dataset, batch_size=1024, shuffle=True, allvec=False):
    has_gloss = dataset.has_gloss
    has_vecs = dataset.has_vecs
    has_electra = dataset.has_electra

    if allvec:
        vec_size = 0
        if has_vecs: vec_size += 2 * data_size_embedding
        if has_electra: vec_size += data_size_embedding
    else: vec_size = data_size_embedding

    def do_collate(json_dicts):
        batch = defaultdict(list)
        for jdict in json_dicts:
            for key in jdict:
                batch[key].append(jdict[key])
        if has_gloss:
            batch["gloss_tensor"] = pad_sequence(
                batch["gloss_tensor"], padding_value=PAD_ix, batch_first=False
            )
        if not allvec:
            if has_vecs:
                for arch in sup_archs:
                    batch[f"{arch}_tensor"] = torch.stack(batch[f"{arch}_tensor"])
            if has_electra:
                batch["electra_tensor"] = torch.stack(batch["electra_tensor"])
        else:
            sgns = torch.stack(batch[f"sgns_tensor"])
            char = torch.stack(batch[f"char_tensor"])
            vecs = [sgns, char]
            if has_electra:
                electra = torch.stack(batch[f"electra_tensor"])
                vecs.insert(0, electra)
            batch["allvec_tensor"] = torch.cat(vecs, dim=1)
        return dict(batch)

    if dataset.has_gloss:
        def do_size_item(item):
            return item["gloss_tensor"].numel()

        return vec_size, DataLoader(
            dataset,
            collate_fn=do_collate,
            batch_sampler=token(
                dataset, batch_size=batch_size, size_fn=do_size_item, shuffle=shuffle
            ),
        )
    else:
        return vec_size, DataLoader(
            dataset, collate_fn=do_collate, batch_size=batch_size, shuffle=shuffle
        )


def check_dataset(dataset, keys):
    if "gloss" in keys:
        assert dataset.has_gloss, "Dataset contains no gloss."
    if "electra" in keys:
        assert dataset.has_electra, "Datatset contains no electra."
    if "sgns" in keys or "char" in keys:
        assert dataset.has_vecs, "Datatset contains no vector."
    if "allvec" in keys:
        assert dataset.has_vecs, "Datatset contains no vector."
    return True


def dataloader(subset, data_file, vocab_lang, vocab_subset, vocab_subdir, vocab_type, vocab_size,
                   input_key, output_key, batch_size=1024, shuffle=False, maxlen=256, lowercase=False):
    if vocab_type == 'sentencepiece':
        vocab = loading_create(vocab_lang, vocab_subset, vocab_subdir, dict_size=vocab_size)
        vocab = SpVocab(vocab)
    else:
        raise ValueError(f'unsupported vocab type: {vocab_type}')
    dataset = data_json(data_file, vocab, maxlen=maxlen, lowercase=lowercase)
    if subset in ['train', 'dev']: check_dataset(dataset, [input_key, output_key])
    elif subset == 'test': check_dataset(dataset, [input_key])
    else:
        raise ValueError(f'unrecognized subset: {subset}')
    return data_loader(dataset, batch_size=batch_size, shuffle=shuffle, allvec=(input_key == 'allvec'))
