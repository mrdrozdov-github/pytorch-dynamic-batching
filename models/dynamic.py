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
        batch_size = len(x)
        lengths = [len(s) for s in x]

        outputs = [Variable(torch.zeros(1, self.model_dim).float(), volatile=not self.training)
                   for _ in range(batch_size)]

        for t in range(max(lengths)):
            batch = []
            h = []
            idx = []
            for i, (s, l) in enumerate(zip(x, lengths)):
                if l >= max(lengths) - t:
                    batch.append(s.pop())
                    h.append(outputs[i])
                    idx.append(i)

            batch = np.concatenate(np.array(batch).reshape(-1, 1), 0)
            emb = Variable(torch.from_numpy(self.initial_embeddings.take(batch, 0)), volatile=not self.training)
            h = torch.cat(h, 0)
            h_next = self.rnn(emb, h)
            h_next = torch.chunk(h_next, len(idx))

            for i, o in zip(idx, h_next):
                outputs[i] = o

        outputs = torch.cat(outputs, 0)
        h = F.relu(self.l0(F.dropout(outputs, 0.5, self.training)))
        h = F.relu(self.l1(F.dropout(h, 0.5, self.training)))
        y = F.log_softmax(h)
        return y


