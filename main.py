import load_sst_data
import utils
import time
import numpy as np

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

training_data_path = 'trees/dev.txt'
eval_data_path = 'trees/dev.txt'
embedding_data_path = 'glove.6B.50d.txt'
word_embedding_dim = 50
model_dim = 100
mlp_dim = 256
batch_size = 32
lr = 0.0001
max_training_steps = 1000
eval_interval_steps = 10
statistics_interval_steps = 10
num_classes = 2

data_manager = load_sst_data

training_data, training_vocab = data_manager.load_data(training_data_path)
eval_data, eval_vocab = data_manager.load_data(eval_data_path)

vocab = set.union(training_vocab, eval_vocab)
vocab = utils.BuildVocabularyForTextEmbeddingFile(
    embedding_data_path, vocab, utils.CORE_VOCABULARY)
initial_embeddings = utils.LoadEmbeddingsFromText(
    vocab, word_embedding_dim, embedding_data_path)

training_data = utils.Tokenize(training_data, vocab)
eval_data = utils.Tokenize(eval_data, vocab)

training_iter = utils.MakeDataIterator(training_data, batch_size, forever=True)()
eval_iter = utils.MakeDataIterator(eval_data, batch_size, forever=False)


def make_batch(examples):
    data = []
    target = []

    for e in examples:
        data.append(list(reversed(e.tokens[:])))
        target.append(e.label)

    return data, target


class Net(nn.Module):
    """Net."""
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
        
    def forward(self, x):
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

model = Net(
model_dim=model_dim,
mlp_dim=mlp_dim,
num_classes=num_classes,
word_embedding_dim=word_embedding_dim,
initial_embeddings=initial_embeddings,
)

optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

print(model)

for step in range(max_training_steps):

    data, target = make_batch(next(training_iter))

    model.train()
    optimizer.zero_grad()
    y = model(data)
    target = torch.LongTensor(target)
    loss = F.nll_loss(y, Variable(target, volatile=False))
    loss.backward()
    optimizer.step()

    pred = y.data.max(1)[1]
    acc = pred.eq(target).sum() / float(batch_size)

    print("Train:", acc, loss.data[0])

    if step % eval_interval_steps == 0:
        for batch in eval_iter():
            pass
        print("eval")
