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
        batch_size, seq_length = x.size()[:2]

        emb = Variable(torch.from_numpy(
            self.initial_embeddings.take(x.numpy(), 0)),
            volatile=not self.training)
        h = Variable(torch.zeros(batch_size, self.model_dim), volatile=not self.training)

        for t in range(seq_length):
            inp = emb[:,t,:]
            h = self.rnn(inp, h)

        h = F.relu(self.l0(F.dropout(h.squeeze(), 0.5, self.training)))
        h = F.relu(self.l1(F.dropout(h, 0.5, self.training)))
        y = F.log_softmax(h)
        return y
