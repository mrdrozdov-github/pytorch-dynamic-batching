import load_sst_data
import utils
import time
import numpy as np

from utils import Accumulator, Args, make_batch

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


args = Args()

# Set args.
args.training_data_path = 'trees/train.txt'
args.eval_data_path = 'trees/dev.txt'
args.embedding_data_path = 'glove.6B.50d.txt'
args.word_embedding_dim = 50
args.model_dim = 100
args.mlp_dim = 256
args.batch_size = 32
args.lr = 0.0001
args.max_training_steps = 50000
args.eval_interval_steps = 100
args.statistics_interval_steps = 100
args.num_classes = 2

# Specify data loader.
data_manager = load_sst_data

# Load data.
training_data, training_vocab = data_manager.load_data(args.training_data_path)
eval_data, eval_vocab = data_manager.load_data(args.eval_data_path)

# Load embeddings.
vocab = set.union(training_vocab, eval_vocab)
vocab = utils.BuildVocabularyForTextEmbeddingFile(
    args.embedding_data_path, vocab, utils.CORE_VOCABULARY)
initial_embeddings = utils.LoadEmbeddingsFromText(
    vocab, args.word_embedding_dim, args.embedding_data_path)

# Tokenize data.
training_data = utils.Tokenize(training_data, vocab)
eval_data = utils.Tokenize(eval_data, vocab)

# Create iterators.
training_iter = utils.MakeDataIterator(training_data, args.batch_size, forever=True)()
eval_iter = utils.MakeDataIterator(eval_data, args.batch_size, forever=False)

# Sentence classification model.
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

# Init model.
model = Net(
    model_dim=args.model_dim,
    mlp_dim=args.mlp_dim,
    num_classes=args.num_classes,
    word_embedding_dim=args.word_embedding_dim,
    initial_embeddings=initial_embeddings,
    )

# Init optimizer.
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

print(model)

A = Accumulator()

# Train loop.
for step in range(args.max_training_steps):

    start = time.time()

    data, target = make_batch(next(training_iter))

    model.train()
    optimizer.zero_grad()
    y = model(data)
    target = torch.LongTensor(target)
    loss = F.nll_loss(y, Variable(target, volatile=False))
    loss.backward()
    optimizer.step()

    pred = y.data.max(1)[1]
    acc = pred.eq(target).sum() / float(args.batch_size)

    end = time.time()

    avg_time = (end - start) / float(args.batch_size)

    A.add('time', avg_time)
    A.add('acc', acc)
    A.add('loss', loss.data[0])

    if step % args.statistics_interval_steps == 0:
        print("Step: {} Acc: {:.5} Loss: {:.5} Time: {:.5}".format(step,
            A.get_avg('acc'),
            A.get_avg('loss'),
            A.get_avg('time'),
            ))

    if step % args.eval_interval_steps == 0:

        accum_acc = []
        accum_loss = []
        accum_time = []

        # Eval loop.
        for batch in eval_iter():
            start = time.time()

            data, target = make_batch(batch)

            model.eval()
            optimizer.zero_grad()
            y = model(data)
            target = torch.LongTensor(target)
            pred = y.data.max(1)[1]
            acc = pred.eq(target).sum() / float(args.batch_size)
            loss = F.nll_loss(y, Variable(target, volatile=False))

            end = time.time()

            avg_time = (end - start) / float(args.batch_size)

            accum_acc.append(acc)
            accum_loss.append(loss.data[0])
            accum_time.append(avg_time)

        end = time.time()
        avg_acc = np.array(accum_acc).mean()
        avg_loss = np.array(accum_loss).mean()
        avg_time = np.array(accum_time).mean()
        print("Step: {} Eval Acc: {:.5} Loss: {:.5} Time: {:.5}".format(step,
            avg_acc,
            avg_loss,
            avg_time,
            ))
