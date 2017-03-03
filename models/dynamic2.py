import numpy as np

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self,
                 model_dim=None,
                 mlp_dim=None,
                 num_classes=None,
                 word_embedding_dim=None,
                 initial_embeddings=None,
                 **kwargs):
        super(Net, self).__init__()
        self.model_dim = model_dim
        self.initial_embeddings = initial_embeddings
        self.rnn = nn.RNNCell(word_embedding_dim, model_dim)
        self.l0 = nn.Linear(model_dim, mlp_dim)
        self.l1 = nn.Linear(mlp_dim, num_classes)
        
    def forward(self, x, lengths):
        batch_size = x.size(0)
        max_len = max(lengths)

        emb = Variable(torch.from_numpy(
            self.initial_embeddings.take(x.numpy(), 0)),
            volatile=not self.training)

        outputs = [Variable(torch.zeros(batch_size, self.model_dim).float(), volatile=not self.training)]

        for t in range(max_len):
            choose = torch.ByteTensor(batch_size)
            indices = []
            not_indices = []
            for i, l in enumerate(lengths):
                if l >= max(lengths) - t:
                    indices.append(i)
                    choose[i] = 1
                else:
                    not_indices.append(i)
                    choose[i] = 0

            # Build batch.
            batch = torch.index_select(emb[:,t,:], 0, Variable(torch.LongTensor(indices), volatile=not self.training))
            h_prev = torch.index_select(outputs[-1], 0, Variable(torch.LongTensor(indices), volatile=not self.training))
            h_next = self.rnn(batch, h_prev)

            # Some preparation for output for next step.
            if len(not_indices) > 0:
                not_h_prev = torch.index_select(outputs[-1], 0, Variable(torch.LongTensor(not_indices), volatile=not self.training))
                _not_h_prev = torch.chunk(not_h_prev, len(not_indices))
            _h_next = torch.chunk(h_next, len(indices))
            
            # Make variable for next step.
            _h = []
            _h_next_idx = 0
            _not_h_prev_idx = 0
            for c in choose:
                if c == 1:
                    _h.append(_h_next[_h_next_idx])
                    _h_next_idx += 1
                else:
                    _h.append(_not_h_prev[_not_h_prev_idx])
                    _not_h_prev_idx += 1
            h = torch.cat(_h, 0)

            outputs.append(h)

        hn = outputs[-1]
        h = F.relu(self.l0(F.dropout(hn, 0.5, self.training)))
        h = F.relu(self.l1(F.dropout(h, 0.5, self.training)))
        y = F.log_softmax(h)
        return y
