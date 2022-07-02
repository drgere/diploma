import torch
import numpy as np
from algo_model.include.encod_position import encod_position
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM, GRU
from vocab_and_glove import vocab


class MLP_input(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1,
                 activation=nn.Tanh, in_dropout=0.1, net_dropout=0.4):

        super(MLP_input, self).__init__()
        assert isinstance(n_layers, int) and n_layers >= 0, f"invalid number of layers: {n_layers}"
        self.in_drop = nn.Dropout(in_dropout)
        self.net_drop = nn.Dropout(net_dropout)
        self.flatten = nn.Flatten()
        if n_layers == 0:
            layers = [nn.Linear(input_size, output_size), activation()]
        else:
            layers = [nn.Linear(input_size, hidden_size), activation(), self.net_drop,
                      nn.Linear(hidden_size, output_size), activation()]
            if n_layers > 1:
                for i in range(n_layers-1):
                    layers.insert(3, self.net_drop)
                    layers.insert(3, activation())
                    layers.insert(3, nn.Linear(hidden_size, hidden_size))
        self.linear_relu_stack = nn.Sequential(*tuple(layers))

    def forward(self, x):
        x = self.in_drop(x)
        logits = self.linear_relu_stack(x)
        return logits


class def_base(nn.Module):
    def __init__(self, vocab_size, d_emb=256, d_input=256, maxlen=256,
                 word_emb=None, pad=vocab.PAD_ix, eos=vocab.EOS_ix, in_dropout=0.1, net_dropout=0.4):
        super(def_base, self).__init__()
        self.name = str(type(self).__name__)
        self.d_emb = d_emb
        self.eos_idx = eos
        self.maxlen = maxlen
        self.d_input = d_input
        self.padding_idx = pad
        self.word_emb = word_emb
        self.vocab_size = vocab_size
        self.in_dropout, self.net_dropout = in_dropout, net_dropout
        self.in_drop, self.net_drop = nn.Dropout(in_dropout), nn.Dropout(net_dropout)

        if d_emb != d_input:
            self.input_adapt = MLP_input(
                input_size=d_input, hidden_size=d_emb, output_size=d_emb, n_layers=0,
                activation=nn.Tanh, in_dropout=0.0, net_dropout=0.2)
        else:
            self.input_adapt = None
        if word_emb is None:
            self.embedding = nn.Embedding(vocab_size, d_emb, padding_idx=self.padding_idx)
        else:
            embs = torch.tensor(word_emb, dtype=torch.float)
            assert (embs.shape[0], embs.shape[1]) == (vocab_size, d_emb) ,\
                    "shape of the pretrained embeddings do not match model specifications"
            self.embedding = nn.Embedding.from_pretrained(embs, freeze=False,
                                                          padding_idx=self.padding_idx)

    def params(self):
        for name, param in self.named_parameters():
            if "embedding" in name and self.word_emb is not None: continue
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
            else:
                nn.init.ones_(param)

    def numParameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def load(file):
        return torch.load(file)

    def save(self, file):
        file.parent.mkdir(exist_ok=True, parents=True)
        torch.save(self, file)


class def_rnn(def_base):

    def __init__(self, vocab_size, d_emb=256, d_input=256, d_hidden=256,
                 n_layers=2, base_arch='gru', use_gateact=True, in_dropout=0.05, net_dropout=0.2, maxlen=256,
                 word_emb=None, pad=vocab.PAD_ix, eos=vocab.EOS_ix, allvec=None, d_allvec=-1):

        super(def_rnn, self).__init__(vocab_size=vocab_size, d_emb=d_emb, d_input=d_input,
                                      maxlen=maxlen, word_emb=word_emb, pad=pad, eos=eos,
                                      in_dropout=in_dropout, net_dropout=net_dropout)
        if base_arch == 'gru': RnnCls = GRU
        elif base_arch == 'lstm': RnnCls = LSTM
        else: raise ValueError(f'unknown rnn architecture: {base_arch}')
        self.d_hidden = d_hidden
        if allvec:
            assert allvec in ('merge', 'concat'), 'allvec must either be "concat" or "merge"'
            assert use_gateact, "No sense using allvec concatenation without gate activation"
            assert d_emb == d_input, "allvec not implemented in combination with input adapt (word emb != input gloss emb)"
            assert d_allvec >= d_emb, "allvec not implemented in case where d_allvec < d_emb"
        self.allvec = allvec
        self.d_allvec = d_allvec
        self._rnn = RnnCls(input_size=d_emb, hidden_size=d_hidden, num_layers=n_layers, dropout=0.0)
        self.base_arch = base_arch
        self.v_proj = nn.Linear(d_hidden, vocab_size)
        self.use_gateact = use_gateact
        if use_gateact: self.weights()
        self.params()

    def weights(self):
        if self.allvec == 'concat':
            d_in = self.d_allvec
        else:
            d_in = self.d_emb
        if self.allvec == 'merge':
            self._Wmerg = nn.Linear(self.d_allvec, self.d_emb)
        self._Wz = nn.Linear(d_in+self.d_hidden, self.d_hidden)
        self._Wr = nn.Linear(d_in+self.d_hidden, d_in)
        self._Wh = nn.Linear(d_in+self.d_hidden, self.d_hidden)
        self._sigmoid = nn.Sigmoid()
        self._tanh = nn.Tanh()

    def layer(self, vectors, rnn_out):

        vectors = self.in_drop(vectors)
        rnn_out = self.net_drop(rnn_out)

        if self.allvec == 'merge':
            vectors = self._tanh(self._Wmerg(vectors))
        vectors = torch.unsqueeze(vectors, 0)
        vecstack = torch.cat([vectors] * len(rnn_out))
        rnn_cat = torch.cat((vecstack, rnn_out), -1)
        zt = self._sigmoid(self._Wz(rnn_cat))
        rt = self._sigmoid(self._Wr(rnn_cat))
        rnn_cat_vecweight = torch.cat((rt * vecstack, rnn_out), -1)
        rnn_out_tilde = self._tanh(self._Wh(rnn_cat_vecweight))
        rnn_out = (1.0 - zt) * rnn_out + zt * rnn_out_tilde
        return rnn_out

    def classifier_word(self, rnn_out):
        rnn_out = self.net_drop(rnn_out)
        word_prob = self.v_proj(rnn_out)
        return word_prob

    def forward(self, vector, input_sequence=None, predict_mode=False):
        device = next(self.parameters()).device
        if vector is not None:
            vector = vector.to(device)
        if input_sequence is not None:
            input_sequence = input_sequence.to(device)
        if not predict_mode:
            embs = self.embedding(input_sequence)
            if self.input_adapt:
                vector = self.input_adapt(vector)
        else:
            vector = input_sequence[0]
            if not self.allvec:
                embs = input_sequence[1:]
            else:
                embs = input_sequence[1:, ..., :self.d_emb]
        if self.allvec:
            input_vector = vector[..., :self.d_input]
        else:
            input_vector = vector
        input_vector = self.in_drop(input_vector)
        embs = self.in_drop(embs)
        seq = torch.cat([input_vector.unsqueeze(0), embs], dim=0)
        rnn_out, _ = self._rnn(seq)
        if self.use_gateact:
            rnn_out = self.layer(vector, rnn_out)
        return self.classifier_word(rnn_out)

    @torch.no_grad()
    def pred_beam(self, vector, decode_fn=None, beam_size=24, debug=False):
        device = next(self.parameters()).device
        vector = vector.to(device)
        if self.input_adapt:
            vector = self.input_adapt(vector)
        beam = None
        logprobs = torch.zeros(beam_size).to(device)
        beam_elements = 1
        if debug:
            print(f'beam: {beam}')
            print(f'logprobs: {logprobs}')
        result = []
        for step_idx in range(self.maxlen):
            if step_idx == 0:
                batch = vector.unsqueeze(0).unsqueeze(0)
                model_out = self(None, batch, True)
            else:
                seqs = beam[:step_idx, :beam_elements]
                init_vecs = torch.cat([vector.unsqueeze(0)] * beam_elements)
                model_out = self(init_vecs, seqs)
            cont_logprobs = F.log_softmax(model_out[-1], dim=-1)
            new_sequences = []
            for b in range(beam_elements):
                for vi in range(self.vocab_size):
                    if vi == self.padding_idx or (step_idx > 0 and vi == vocab.BOS_ix): continue
                    new_sequences.append((b, vi, logprobs[b]+cont_logprobs[b, vi]))
            if debug:
                print(f'new_sequences len: {len(new_sequences)}')
            invlp = np.array([-(ns[2].cpu()) for ns in new_sequences])
            sort_ix = np.argsort(invlp)
            new_sequences = [ new_sequences[si] for si in sort_ix ]
            if beam_size < len(new_sequences):
                # take top beam size
                new_sequences = new_sequences[:beam_size]
            beam_elements = len(new_sequences)
            new_beam = torch.zeros(step_idx+2, beam_size, dtype=int).to(device)
            bi = 0
            for ns in new_sequences:
                parent_ix, cont_ix, logprob = ns
                if cont_ix != self.eos_idx:
                    for wi in range(step_idx):
                        new_beam[wi, bi] = beam[wi, parent_ix]
                    new_beam[step_idx, bi] = cont_ix
                    logprobs[bi] = logprob
                    bi += 1
                else:
                    res_seq = beam[:, parent_ix]
                    res_seq[step_idx] = self.eos_idx
                    result.append((res_seq, logprob))
                    beam_size -= 1
                    beam_elements -= 1
            if beam_size == 0: break
            beam = new_beam
            if debug:
                print(f'beam: {beam}')
                print('sequences in the beam:')
                for i in range(beam_elements):
                    print(decode_fn(beam[:, i]))
                print()
        if len(result) == 0:
            beam[self.maxlen, 0] = self.eos_idx
            res_seq = beam[:, 0]
            result.append((res_seq, -1))
        if debug:
            print('RESULT:')
            for re in result:
                seq, lp = re
                print(seq)
                print(decode_fn(seq), f'{lp}')
        return result[0][0]

    @torch.no_grad()
    def pred(self, vector, decode_fn=None, beam_size=64, verbose=False):
        device = next(self.parameters()).device
        batch_size = vector.size(0)
        vector = vector.to(device)
        if self.input_adapt:
            vector = torch.stack([self.input_adapt(row) for row in vector])
        generated_symbols = torch.zeros(0, batch_size, beam_size, dtype=torch.long).to(device)
        current_beam_size = 1
        has_stopped = torch.tensor([False] * (batch_size * current_beam_size)).to(device)
        vector_src = vector.unsqueeze(1).expand(batch_size, current_beam_size, -1).reshape(1,  batch_size * current_beam_size, -1)
        src = vector_src
        logprobs = torch.zeros(batch_size, current_beam_size, dtype=torch.double).to(device)
        lengths = torch.zeros(batch_size * current_beam_size, dtype=torch.int).to(device)
        for step_idx in range(self.maxlen):
            v_dist = self(None, src, True)[-1]
            v_dist[..., self.padding_idx] = -float("inf")
            v_dist = F.log_softmax(v_dist, dim=-1)
            new_logprobs, new_symbols = v_dist.topk(beam_size, dim=-1)
            new_logprobs = new_logprobs.masked_fill(has_stopped.unsqueeze(-1), 0.0)
            lengths += (~has_stopped).int()

            logprobs_ = logprobs.view(batch_size * current_beam_size, 1).expand(batch_size * current_beam_size, beam_size)
            logprobs_ = logprobs_ + new_logprobs
            avg_logprobs = logprobs_
            avg_logprobs, selected_beams = avg_logprobs.view(batch_size, current_beam_size * beam_size).topk(beam_size, dim=-1)
            logprobs = logprobs_.view(batch_size, current_beam_size * beam_size).gather(-1, selected_beams).view(batch_size, beam_size)

            generated_symbols_ = generated_symbols.view(-1, batch_size * current_beam_size, 1).expand(-1, batch_size * current_beam_size, beam_size)
            generated_symbols_ = torch.cat([generated_symbols_, new_symbols.unsqueeze(0)], dim=0)
            generated_symbols_ = generated_symbols_.view(-1, batch_size, current_beam_size * beam_size)
            generated_symbols = generated_symbols_.gather(-1, selected_beams.unsqueeze(0).expand(step_idx + 1, batch_size,  beam_size)).view(step_idx + 1, batch_size, beam_size)

            lengths = lengths.view(batch_size, current_beam_size, 1).expand(batch_size, current_beam_size, beam_size)
            lengths = lengths.reshape(batch_size, current_beam_size * beam_size).gather(-1, selected_beams).view(-1)
            has_stopped = has_stopped.view(batch_size, current_beam_size, 1).expand(batch_size, current_beam_size, beam_size)
            has_stopped = has_stopped.reshape(batch_size, current_beam_size * beam_size).gather(-1, selected_beams).view(-1)

            generated_symbols = generated_symbols.view(-1, batch_size * beam_size)
            generated_symbols[-1] = generated_symbols[-1].masked_fill(has_stopped, self.padding_idx)
            has_stopped = has_stopped | (generated_symbols.view(-1, batch_size * beam_size)[-1] == self.eos_idx).view(batch_size * beam_size)

            orig_embs = self.embedding(generated_symbols)
            if self.allvec:
                embs = F.pad(input=orig_embs, pad=(0, self.d_allvec - self.d_emb),
                             mode='constant', value=0)
            else: embs = orig_embs
            src = torch.cat([vector_src.expand(1, beam_size, -1), embs], dim=0)
            generated_symbols = generated_symbols.view(-1, batch_size, beam_size)

            if has_stopped.all():
                break
            current_beam_size = beam_size

        max_scores, selected_beams = (logprobs / lengths.view(batch_size, beam_size)).topk(1, dim=1)
        output_sequence = generated_symbols.gather(1, selected_beams.unsqueeze(0).expand(step_idx + 1, batch_size, 1))
        if verbose:
            print(decode_fn(output_sequence.squeeze(-1)))
        return output_sequence.squeeze(-1)


class defmod_transformer(def_base):

    def __init__(
        self, vocab_size, d_emb=256, d_input=256, n_head=4, n_layers=4, dropout=0.3, maxlen=256,
            word_emb=None, pad=vocab.PAD_ix, eos=vocab.EOS_ix
    ):
        super(defmod_transformer, self).__init__(vocab_size=vocab_size, d_emb=d_emb, d_input=d_input,
                                        maxlen=maxlen, word_emb=word_emb, pad=pad, eos=eos)
        self.positional_encoding = encod_position(d_emb, dropout=dropout, max_len=maxlen)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_emb, nhead=n_head, dropout=dropout, dim_feedforward=d_emb * 2
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.v_proj = nn.Linear(d_emb, vocab_size)
        self.params()

    def subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, vector, input_sequence=None):
        device = next(self.parameters()).device
        embs = self.embedding(input_sequence)
        if self.input_adapt: vector = self.input_adapt(vector)
        seq = torch.cat([vector.unsqueeze(0), embs], dim=0)
        src = self.positional_encoding(seq)
        src_mask = self.subsequent_mask(src.size(0)).to(device)
        src_key_padding_mask = torch.cat(
            [
                torch.tensor([[False] * input_sequence.size(1)]).to(device),
                (input_sequence == self.padding_idx),
            ],
            dim=0,
        ).t()
        transformer_output = self.transformer_encoder(
            src, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )
        v_dist = self.v_proj(transformer_output)
        return v_dist

    @torch.no_grad()
    def pred(self, vector, decode_fn=None, beam_size=64, verbose=False):
        device = next(self.parameters()).device
        vector = vector.to(device)
        batch_size = vector.size(0)
        if self.input_adapt:
            vector = torch.stack([self.input_adapt(row) for row in vector])

        generated_symbols = torch.zeros(0, batch_size, beam_size, dtype=torch.long).to(device)

        current_beam_size = 1
        has_stopped = torch.tensor([False] * (batch_size * current_beam_size)).to(device)

        vector_src = vector.unsqueeze(1).expand(batch_size, current_beam_size, -1).reshape(1,  batch_size * current_beam_size, -1)
        src = vector_src
        src_key_padding_mask = torch.tensor([[False] * (batch_size * current_beam_size)]).to(device)

        logprobs = torch.zeros(batch_size, current_beam_size, dtype=torch.double).to(device)
        lengths = torch.zeros(batch_size * current_beam_size, dtype=torch.int).to(device)
        for step_idx in range(self.maxlen):

            src_mask = self.subsequent_mask(src.size(0)).to(device)
            src_pe = self.positional_encoding(src)
            transformer_output = self.transformer_encoder(
                src_pe, mask=src_mask, src_key_padding_mask=src_key_padding_mask.t()
            )[-1]
            v_dist = self.v_proj(transformer_output)
            v_dist[...,self.padding_idx] = -float("inf")
            v_dist = F.log_softmax(v_dist, dim=-1)

            new_logprobs, new_symbols = v_dist.topk(beam_size, dim=-1)
            new_logprobs = new_logprobs.masked_fill(has_stopped.unsqueeze(-1), 0.0)
            lengths += (~has_stopped).int()

            logprobs_ = logprobs.view(batch_size * current_beam_size, 1).expand(batch_size * current_beam_size, beam_size)
            logprobs_ = logprobs_ + new_logprobs
            avg_logprobs = logprobs_
            avg_logprobs, selected_beams = avg_logprobs.view(batch_size, current_beam_size * beam_size).topk(beam_size, dim=-1)
            logprobs = logprobs_.view(batch_size, current_beam_size * beam_size).gather(-1, selected_beams).view(batch_size, beam_size)

            generated_symbols_ = generated_symbols.view(-1, batch_size * current_beam_size, 1).expand(-1, batch_size * current_beam_size, beam_size)
            generated_symbols_ = torch.cat([generated_symbols_, new_symbols.unsqueeze(0)], dim=0)
            generated_symbols_ = generated_symbols_.view(-1, batch_size, current_beam_size * beam_size)
            generated_symbols = generated_symbols_.gather(-1, selected_beams.unsqueeze(0).expand(step_idx + 1, batch_size,  beam_size)).view(step_idx + 1, batch_size, beam_size)

            lengths = lengths.view(batch_size, current_beam_size, 1).expand(batch_size, current_beam_size, beam_size)
            lengths = lengths.reshape(batch_size, current_beam_size * beam_size).gather(-1, selected_beams).view(-1)
            has_stopped = has_stopped.view(batch_size, current_beam_size, 1).expand(batch_size, current_beam_size, beam_size)
            has_stopped = has_stopped.reshape(batch_size, current_beam_size * beam_size).gather(-1, selected_beams).view(-1)

            generated_symbols = generated_symbols.view(-1, batch_size * beam_size)
            generated_symbols[-1] = generated_symbols[-1].masked_fill(has_stopped, self.padding_idx)
            has_stopped = has_stopped | (generated_symbols.view(-1, batch_size * beam_size)[-1] == self.eos_idx).view(batch_size * beam_size)

            src_key_padding_mask = src_key_padding_mask.view(-1, batch_size, current_beam_size, 1).expand(-1, batch_size, current_beam_size, beam_size)
            src_key_padding_mask = src_key_padding_mask.reshape(-1, batch_size, current_beam_size * beam_size)
            src_key_padding_mask = src_key_padding_mask.gather(-1, selected_beams.unsqueeze(0).expand(step_idx + 1, batch_size,  beam_size)).view(step_idx + 1, batch_size * beam_size)
            src_key_padding_mask = torch.cat([src_key_padding_mask, has_stopped.unsqueeze(0)], dim=0)

            src = torch.cat([vector_src.expand(1, beam_size, -1), self.embedding(generated_symbols)], dim=0)
            generated_symbols = generated_symbols.view(-1, batch_size, beam_size)

            if has_stopped.all():
                break
            current_beam_size = beam_size

        max_scores, selected_beams = (logprobs / lengths.view(batch_size, beam_size)).topk(1, dim=1)
        output_sequence = generated_symbols.gather(1, selected_beams.unsqueeze(0).expand(step_idx + 1, batch_size, 1))
        if verbose:
            print(decode_fn(output_sequence.squeeze(-1)))
        return output_sequence.squeeze(-1)
