LABEL_MAP = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
}


class Example(object):
    label = None
    sentence = None
    tokens = None
    transitions = None

    def __repr__(self):
        return str(self.__dict__)
        

def convert_unary_binary_bracketed_data(filename, binary=False):
    # Build a binary tree out of a binary parse in which every
    # leaf node is wrapped as a unary constituent, as here:
    #   (4 (2 (2 The ) (2 actors ) ) (3 (4 (2 are ) (3 fantastic ) ) (2 . ) ) )
    examples = []
    vocab = set()
    with open(filename, 'r') as f:
        for line in f:
            example = Example()
            line = line.strip()
            line = line.replace(')', ' )')
            if len(line) == 0:
                continue
            example.label = int(line[1])

            if binary:
                label = example.label
                if label < 2:
                    example.label = 0
                elif label > 2:
                    example.label = 1
                else:
                    continue

            example.sentence = line
            example.tokens = []
            example.transitions = []

            words = example.sentence.split(' ')
            for index, word in enumerate(words):
                if word[0] != "(":
                    if word == ")":  
                        # Ignore unary merges
                        if words[index - 1] == ")":
                            example.transitions.append(1)
                    else:
                        # Downcase all words to match GloVe.
                        w = word.lower()
                        example.tokens.append(w)
                        vocab.add(w)
                        example.transitions.append(0)
            examples.append(example)
    return examples, vocab

def load_data(filename):
    return convert_unary_binary_bracketed_data(filename, binary=True)

if __name__ == '__main__':
    import sys

    def get(l, idx, default):
        try:
            return l[idx]
        except IndexError:
            return default

    path = get(sys.argv, 1, 'trees/dev.txt')
    binary = int(get(sys.argv, 2, '0')) == 1
    examples, vocab = convert_unary_binary_bracketed_data(path, binary)
    print(examples[0])
    print(len(vocab))
