import load_sst_data
import utils
import time

import torch

training_data_path = 'trees/dev.txt'
eval_data_path = 'trees/dev.txt'
embedding_data_path = 'glove.6B.50d.txt'
word_embedding_dim = 50
batch_size = 32
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
    lengths = []

    for e in examples:
        data.append(list(reversed(e.tokens[:])))
        target.append(e.label)
        lengths.append(len(e.tokens))

    return data, target, lengths


for step in range(max_training_steps):

    data, target, lengths = make_batch(next(training_iter))

    print(data[0])
    print(data[1])

    outputs = {i: None for i in range(batch_size)}

    for t in range(max(lengths)):
        batch = []
        h = []
        idx = []
        for i, (s, l) in enumerate(zip(data, lengths)):
            if l >= max(lengths) - t:
                batch.append(s.pop())
                h.append(outputs[i])
                idx.append(i)

        # batch = torch.cat(batch, 0)
        # h = torch.cat(h, 0)
        # h_next = model(batch, h)
        # h_next = torch.chunk(h_next, len(idx))

        # TODO: Remove.
        h_next = range(len(idx))

        for i, o in zip(idx, h_next):
            outputs[i] = o

        print(batch)
        print(len(batch))

    time.sleep(0.1)

    print("train")

    break

    if step % eval_interval_steps == 0:
        for batch in eval_iter():
            time.sleep(0.1)
        print("eval")