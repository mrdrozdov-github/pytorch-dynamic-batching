import unittest

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

import models
import load_sst_data
import utils


def default_args(style="dynamic"):
    args = utils.Args()

    # Set args.
    args.style = style
    args.smart_batching = True
    args.training_data_path = 'test_trees.txt'
    args.embedding_data_path = 'test_embeddings.txt'
    args.word_embedding_dim = 50
    args.model_dim = 100
    args.mlp_dim = 256
    args.batch_size = 2
    args.lr = 0.0001
    args.max_training_steps = 50000
    args.eval_interval_steps = 100
    args.statistics_interval_steps = 100
    args.num_classes = 2

    return args


def model_suite(self, mdl):
    data_iter = self.data_iter
    args = self.args
    initial_embeddings = self.initial_embeddings

    args.style = mdl

    # Model Class.
    model_cls = getattr(models, mdl)

    # Test model.
    model = model_cls(
        model_dim=args.model_dim,
        mlp_dim=args.mlp_dim,
        num_classes=args.num_classes,
        word_embedding_dim=args.word_embedding_dim,
        initial_embeddings=initial_embeddings,
        )

    data, target, lengths = utils.make_batch(next(data_iter), args.style == "dynamic")
    y = model(data, lengths)


class ModelsTestCase(unittest.TestCase):

    def setUp(self):
        args = default_args()

        # Specify data loader.
        data_manager = load_sst_data

        # Load data.
        raw_data, vocab = data_manager.load_data(args.training_data_path)

        # Load embeddings.
        vocab = utils.BuildVocabularyForTextEmbeddingFile(
            args.embedding_data_path, vocab, utils.CORE_VOCABULARY)
        initial_embeddings = utils.LoadEmbeddingsFromText(
            vocab, args.word_embedding_dim, args.embedding_data_path)

        # Tokenize data.
        tokenized = utils.Tokenize(raw_data, vocab)

        # Create iterators.
        data_iter = utils.MakeDataIterator(tokenized, args.batch_size, smart_batching=args.smart_batching, forever=True)()

        # Cache useful values.
        self.args = args
        self.data_iter = data_iter
        self.initial_embeddings = initial_embeddings

    def test_dynamic(self):
        mdl = 'dynamic'
        model_suite(self, mdl)

    def test_dynamic2(self):
        mdl = 'dynamic2'
        model_suite(self, mdl)

    def test_static(self):
        mdl = 'static'
        model_suite(self, mdl)

    def test_static2(self):
        mdl = 'static2'
        model_suite(self, mdl)

    def test_fakedynamic(self):
        mdl = 'fakedynamic'
        model_suite(self, mdl)

    def test_fakestatic(self):
        mdl = 'fakestatic'
        model_suite(self, mdl)


if __name__ == '__main__':
    unittest.main()
