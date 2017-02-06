import numpy as np
import random
from collections import deque


# With loaded embedding matrix, the padding vector will be initialized to zero
# and will not be trained. Hopefully this isn't a problem. It seems better than
# random initialization...
PADDING_TOKEN = "*PADDING*"

# Temporary hack: Map UNK to "_" when loading pretrained embedding matrices:
# it's a common token that is pretrained, but shouldn't look like any content words.
UNK_TOKEN = "_"


CORE_VOCABULARY = {PADDING_TOKEN: 0,
                   UNK_TOKEN: 1}


def BuildVocabularyForTextEmbeddingFile(path, types_in_data, core_vocabulary):
    """Quickly iterates through a GloVe-formatted text vector file to
    extract a working vocabulary of words that occur both in the data and
    in the vector file."""

    vocabulary = {}
    vocabulary.update(core_vocabulary)
    next_index = len(vocabulary)
    with open(path, 'rU') as f:
        for line in f:
            spl = line.split(" ", 1)
            word = unicode(spl[0].decode('UTF-8'))
            if word in types_in_data and word not in vocabulary:
                vocabulary[word] = next_index
                next_index += 1
    return vocabulary


def LoadEmbeddingsFromText(vocabulary, embedding_dim, path):
    """Prepopulates a numpy embedding matrix indexed by vocabulary with
    values from a GloVe - format vector file.

    For now, values not found in the file will be set to zero."""
    
    emb = np.zeros(
        (len(vocabulary), embedding_dim), dtype=np.float32)
    with open(path, 'r') as f:
        for line in f:
            spl = line.split(" ")
            word = spl[0]
            if word in vocabulary:
                emb[vocabulary[word], :] = [float(e) for e in spl[1:]]
    return emb


def MakeDataIterator(examples, batch_size, forever=True, smart_batching=True, num_buckets=10):
    if smart_batching:
        def data_iter():
            def build_bucketed_batch_indices():
                batches = []
                lengths = [(i, len(e.tokens)) for i, e in enumerate(examples)]

                # Shuffle before bucketing.
                random.shuffle(lengths)

                bucket_size = len(examples) // num_buckets
                buckets = [lengths[i*bucket_size:(i+1)*bucket_size] for i in range(num_buckets)]
                buckets = [sorted(b, key=lambda x: x[1]) for b in buckets]
                for b in buckets:
                    num_batches = len(b) // batch_size
                    for i in range(num_batches):
                        _batch = b[i*batch_size:(i+1)*batch_size]
                        _batch = [x[0] for x in _batch]
                        batches.append(_batch)

                # Shuffle after bucketing
                random.shuffle(batches)

                return batches

            batch_indices = build_bucketed_batch_indices()
            num_batches = len(batch_indices)
            start = -1

            while True:
                start += 1
                if start >= num_batches:

                    if not forever:
                        break

                    # Start another epoch.
                    batch_indices = build_bucketed_batch_indices()
                    num_batches = len(batch_indices)
                    start = 0
                yield tuple(examples[i] for i in batch_indices[start])

    else:
        def data_iter():
            dataset_size = len(examples)
            start = -1 * batch_size
            order = range(dataset_size)
            random.shuffle(order)

            while True:
                start += batch_size
                if start > dataset_size - batch_size:

                    if not forever:
                        break

                    # Start another epoch.
                    start = 0
                    random.shuffle(order)
                batch_indices = order[start:start + batch_size]
                yield tuple(examples[i] for i in batch_indices)

    return data_iter


def Tokenize(examples, vocabulary):
    for e in examples:
        e.tokens = [vocabulary.get(w, vocabulary.get(UNK_TOKEN)) for w in e.tokens]
    return examples


class Accumulator(object):

    cache = dict()

    def __init__(self, trail=100):
        self.trail = trail

    def add(self, key, val):
        self.cache.setdefault(key, deque(maxlen=self.trail)).append(val)

    def get(self, key):
        ret = self.cache.get(key, [])
        try:
            del self.cache[key]
        except:
            pass
        return ret

    def get_avg(self, key):
        return np.array(self.get(key)).mean()


class Args(object):
    def __repr__(self):
        s = "{}"
        return s.format(self.__dict__)
