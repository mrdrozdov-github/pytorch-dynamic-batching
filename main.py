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

import models
from utils import Accumulator, Args, make_batch

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

# Pick model.
Net = getattr(models, args.style)

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
