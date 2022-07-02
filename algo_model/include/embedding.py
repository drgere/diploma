import torch
import torch.nn as nn
import torch.nn.functional as F


class embedding(nn.Module):

    def __init__(self, d_model=256, n_layers=1, dropout=0.3, **kwargs):
        super(embedding, self).__init__()
        self.name = str(type(self).__name__)
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.n_layers = n_layers
        d_hidden = d_model

        self.layer_start = nn.Linear(d_model, d_hidden)
        self.layer1 = nn.Linear(d_hidden, d_hidden)
        self.layer2 = nn.Linear(d_hidden, d_hidden)
        self.layer3 = nn.Linear(d_hidden, d_hidden)
        self.layer4 = nn.Linear(d_hidden, d_hidden)
        self.layer_end = nn.Linear(d_hidden, d_model)
        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        self.layers = [ self.layers[i] for i in range(n_layers-1)]
        self.layers = [self.layer_start] + self.layers + [self.layer_end]

        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
            else:
                nn.init.ones_(param)

    def forward(self, vector):
        x = vector
        for l in self.layers:
            x = self.dropout(F.relu(l(x)))
        return x

    @staticmethod
    def load(file):
        return torch.load(file)

    def save(self, file):
        torch.save(self, file)