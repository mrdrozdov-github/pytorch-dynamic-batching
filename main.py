import load_sst_data
import utils
import time

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

for step in range(max_training_steps):

    batch = next(training_iter)

    time.sleep(0.1)

    print("train")

    if step % eval_interval_steps == 0:
        for batch in eval_iter():
            time.sleep(0.1)
        print("eval")