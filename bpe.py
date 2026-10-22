import re, collections


def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


def build_word_vocab(infile):
    """Count words in the training file, written as space-separated chars + '</w>'."""
    counts = collections.Counter()
    with open(infile, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                # Same as sentiment_data.py: "<label>\t<sentence>", lowercased
                sent = line.split("\t", 1)[-1].lower()
                counts.update(sent.split())
    return {' '.join(word) + ' </w>': freq for word, freq in counts.items()}


def learn_bpe(infile, vocab_size):
    """Keep merging the most frequent pair until the subword vocab hits vocab_size."""
    vocab = build_word_vocab(infile)

    # Start with every single character (plus '</w>') as a token.
    tokens = set()
    for word in vocab:
        tokens.update(word.split())

    merges = []
    while len(tokens) < vocab_size:
        pairs = get_stats(vocab)
        if not pairs:  # every word is already a single token
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        merges.append(best)
        tokens.add(''.join(best))  # each merge adds exactly one new token
    return merges, sorted(tokens)


def encode_word(word, merges):
    """Split a new word into subwords by replaying the learned merges in order."""
    symbols = list(word) + ['</w>']
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols[i:i + 2] = [a + b]
            else:
                i += 1
    return symbols


def encode_sentence(words, merges):
    return [tok for w in words for tok in encode_word(w, merges)]


class BPETokenizer:
    """Learned BPE merges plus a token -> index map (PAD = 0, UNK = 1)."""

    def __init__(self, infile, vocab_size):
        self.merges, tokens = learn_bpe(infile, vocab_size)
        self.tok2idx = {"PAD": 0, "UNK": 1}
        for t in tokens:
            self.tok2idx[t] = len(self.tok2idx)
        self._cache = {}

    def __len__(self):
        return len(self.tok2idx)

    def encode(self, words):
        """Turn a list of words into a list of subword indices."""
        idxs = []
        for w in words:
            if w not in self._cache:
                self._cache[w] = [self.tok2idx.get(t, 1) for t in encode_word(w, self.merges)]
            idxs.extend(self._cache[w])
        return idxs


if __name__ == "__main__":
    vocab_size = 10000
    merges, tokens = learn_bpe("data/train.txt", vocab_size)
    print(f"Learned {len(merges)} merges, vocab size = {len(tokens)}")
    print("First 10 merges:", merges[:10])
    print(encode_sentence("the movie was unbelievably good".split(), merges))
