"""

Dynamic Batched RNNs
====================

A. Static - Straightforward RNN.
B. Static2 - RNN using an RNNCell in a loop.
C. Dynamic - RNN that has dynamically sized input at each timestep. No padding.
D. Dynamic2 - RNN that has dynamically sized input at each timestep. Uses padding.


Speed Ranking (CPU)
=============

1. Static/Static2 - Similar performance. Surprisingly faster than dynamic alternatives.
2. Dynamic
3. Dynamic2

"""

import load_sst_data
import pprint
import utils
import time
import sys
import numpy as np
import gflags

from utils import Accumulator, Args

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


FLAGS = gflags.FLAGS
args = Args()

gflags.DEFINE_enum("style", "dynamic", ["static", "static2", "dynamic", "dynamic2", "fakedynamic", "fakestatic"],
    "Specify dynamic or static RNN loops.")
gflags.DEFINE_boolean("smart_batching", True, "Bucket batches for similar length.")

# Parse command line flags.
FLAGS(sys.argv)

# Set args.
args.training_data_path = 'trees/dev.txt'
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

args.__dict__.update(FLAGS.FlagValuesDict())

pp = pprint.PrettyPrinter(indent=4)

print("Args: {}".format(pp.pformat(args.__dict__)))

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
training_iter = utils.MakeDataIterator(training_data, args.batch_size, smart_batching=args.smart_batching, forever=True)()
eval_iter = utils.MakeDataIterator(eval_data, args.batch_size, smart_batching=args.smart_batching, forever=False)

# Sentence classification models.
class DynamicNet(nn.Module):
    """DynamicNet."""
    def __init__(self,
                 model_dim=None,
                 mlp_dim=None,
                 num_classes=None,
                 word_embedding_dim=None,
                 initial_embeddings=None,
                 **kwargs):
        super(DynamicNet, self).__init__()
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


class DynamicNet2(nn.Module):
    """DynamicNet2."""
    def __init__(self,
                 model_dim=None,
                 mlp_dim=None,
                 num_classes=None,
                 word_embedding_dim=None,
                 initial_embeddings=None,
                 **kwargs):
        super(DynamicNet2, self).__init__()
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


class FakeDynamicNet(nn.Module):
    """FakeDynamicNet."""
    def __init__(self,
                 model_dim=None,
                 mlp_dim=None,
                 num_classes=None,
                 word_embedding_dim=None,
                 initial_embeddings=None,
                 **kwargs):
        super(FakeDynamicNet, self).__init__()
        self.word_embedding_dim = word_embedding_dim
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

        for t in range(max_len):
            indices = []
            for i, l in enumerate(lengths):
                if l >= max(lengths) - t:
                    indices.append(i)

            # Build batch.
            dynamic_batch_size = len(indices)
            inp = Variable(torch.FloatTensor(dynamic_batch_size, self.word_embedding_dim), volatile=not self.training)
            h = Variable(torch.FloatTensor(dynamic_batch_size, self.model_dim), volatile=not self.training)
            output = self.rnn(inp, h)

        hn = output
        h = F.relu(self.l0(F.dropout(hn.squeeze(), 0.5, self.training)))
        h = F.relu(self.l1(F.dropout(h, 0.5, self.training)))
        y = F.log_softmax(h)
        return y


class FakeStaticNet(nn.Module):
    """FakeStaticNet."""
    def __init__(self,
                 model_dim=None,
                 mlp_dim=None,
                 num_classes=None,
                 word_embedding_dim=None,
                 initial_embeddings=None,
                 **kwargs):
        super(FakeStaticNet, self).__init__()
        self.word_embedding_dim = word_embedding_dim
        self.model_dim = model_dim
        self.initial_embeddings = initial_embeddings
        self.rnn = nn.RNN(word_embedding_dim, model_dim, batch_first=True)
        self.l0 = nn.Linear(model_dim, mlp_dim)
        self.l1 = nn.Linear(mlp_dim, num_classes)
        
    def forward(self, x, lengths):
        batch_size = x.size(0)
        max_len = max(lengths)

        emb = Variable(torch.from_numpy(
            self.initial_embeddings.take(x.numpy(), 0)),
            volatile=not self.training)
        inp = Variable(torch.FloatTensor(emb.size()), volatile=not self.training)
        h0 = Variable(torch.FloatTensor(1, batch_size, self.model_dim), volatile=not self.training)

        _, hn = self.rnn(emb, h0)

        h = F.relu(self.l0(F.dropout(hn.squeeze(), 0.5, self.training)))
        h = F.relu(self.l1(F.dropout(h, 0.5, self.training)))
        y = F.log_softmax(h)
        return y



class StaticNet(nn.Module):
    """StaticNet."""
    def __init__(self,
                 model_dim=None,
                 mlp_dim=None,
                 num_classes=None,
                 word_embedding_dim=None,
                 initial_embeddings=None,
                 **kwargs):
        super(StaticNet, self).__init__()
        self.model_dim = model_dim
        self.initial_embeddings = initial_embeddings
        self.rnn = nn.RNN(word_embedding_dim, model_dim, batch_first=True)
        self.l0 = nn.Linear(model_dim, mlp_dim)
        self.l1 = nn.Linear(mlp_dim, num_classes)
        
    def forward(self, x, lengths):
        batch_size = x.size(0)

        emb = Variable(torch.from_numpy(
            self.initial_embeddings.take(x.numpy(), 0)),
            volatile=not self.training)
        h0 = Variable(torch.zeros(1, batch_size, self.model_dim), volatile=not self.training)

        _, hn = self.rnn(emb, h0)

        h = F.relu(self.l0(F.dropout(hn.squeeze(), 0.5, self.training)))
        h = F.relu(self.l1(F.dropout(h, 0.5, self.training)))
        y = F.log_softmax(h)
        return y


class StaticNet2(nn.Module):
    """StaticNet2."""
    def __init__(self,
                 model_dim=None,
                 mlp_dim=None,
                 num_classes=None,
                 word_embedding_dim=None,
                 initial_embeddings=None,
                 **kwargs):
        super(StaticNet2, self).__init__()
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

# Pick model.
if args.style == "dynamic":
    Net = DynamicNet
elif args.style == "dynamic2":
    Net = DynamicNet2
elif args.style == "static":
    Net = StaticNet
elif args.style == "static2":
    Net = StaticNet2
elif args.style == "fakedynamic":
    Net = FakeDynamicNet
elif args.style == "fakestatic":
    Net = FakeStaticNet
else:
    raise NotImplementedError

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
print("Total Params: {}".format(sum(torch.numel(p.data) for p in model.parameters())))

A = Accumulator()

def make_batch(examples, dynamic=True):
    # Build lengths.
    lengths = []
    for e in examples:
        lengths.append(len(e.tokens))

    # Build input.
    if dynamic: # dynamic: list of lists
        data = []
        for e in examples:
            d = list(reversed(e.tokens[:]))
            data.append(d)
    else: # static: batch matrix
        batch_size = len(examples)
        max_len = max(len(e.tokens) for e in examples)
        data = torch.zeros(batch_size, max_len).long()
        for i, e in enumerate(examples):
            l = len(e.tokens)
            offset = max_len - l
            data[i,offset:max_len] = torch.Tensor(e.tokens[:])

    # Build labels.
    target = []
    for e in examples:
        target.append(e.label)
    target = torch.LongTensor(target)

    return data, target, lengths

# Train loop.
for step in range(args.max_training_steps):

    start = time.time()

    data, target, lengths = make_batch(next(training_iter), args.style == "dynamic")

    model.train()
    optimizer.zero_grad()
    y = model(data, lengths)
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

            data, target, lengths = make_batch(batch, args.style == "dynamic")

            model.eval()
            optimizer.zero_grad()
            y = model(data, lengths)
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
