# CSE256 PA1: Sentiment Classification

## Setup

Requires Python 3 with these packages:

```
pip install torch scikit-learn matplotlib numpy
```

Run all commands from this directory, because the data paths (`data/...`) are relative.

## Commands

| Command | What it does |
|---|---|
| `python main.py --model BOW` | Trains 2-layer and 3-layer bag-of-words networks. Saves `train_accuracy.png` and `dev_accuracy.png`. |
| `python main.py --model DAN` | Trains 2-layer and 3-layer Deep Averaging Networks over the GloVe word vocabulary. Saves `dan_train_accuracy.png` and `dan_dev_accuracy.png`. |
| `python main.py --model DAN-BPE --vocab_size 2000` | Learns a byte-pair encoding (BPE) subword vocabulary of the given size from `train.txt`, then trains 2-layer and 3-layer DANs on subwords with randomly initialized embeddings. `--vocab_size` defaults to 1000. Saves `dan_bpe<size>_train_accuracy.png` and `dan_bpe<size>_dev_accuracy.png`. |
| `python bpe.py` | Learns BPE with vocab size 1000 on `train.txt` and prints the merge count, the first 10 merges, and a sample sentence split into subwords. |
| `python sentiment_data.py <word1> <word2>` | Prints the cosine similarity of two words using the 50d GloVe embeddings. |

BOW models train for 100 epochs (lr = 1e-4, batch size 16). DAN and DAN-BPE models train for 30 epochs (lr = 1e-3, batch size 64). Both print train and dev accuracy every 10 epochs.

## Code overview

- **`main.py`**: Entry point.
  - `train_epoch` and `eval_epoch` run one pass over a `DataLoader` and return accuracy and average loss.
  - `experiment` trains a model with NLL loss and Adam, and records the accuracy after each epoch. The learning rate and number of epochs are arguments (defaults: lr = 1e-4, 100 epochs).
  - `main` reads `--model` and `--vocab_size`, builds the datasets and models, runs the experiments, and saves the plots. `DAN` and `DAN-BPE` share the same training and plotting code; they differ only in the dataset and in how the model is built.
- **`BOWmodels.py`**
  - `SentimentDatasetBOW` turns each sentence into a 512-dimensional word-count vector with sklearn's `CountVectorizer`. The dev set reuses the vocabulary fitted on the training set.
  - `NN2BOW` and `NN3BOW` are feedforward networks with ReLU hidden layers and a 2-class log-softmax output.
- **`DANmodels.py`**
  - `SentimentDatasetDAN` maps each word to its index in the GloVe vocabulary. Unknown words get the `UNK` index (1). Sentences are padded with `PAD` (0) to the length of the longest sentence.
  - `SentimentDatasetDANBPE` does the same with BPE subwords: each sentence becomes the concatenated subword indices of its words (from `BPETokenizer.encode`), padded with 0.
  - `DAN` looks up embeddings, averages them over the real tokens, then applies `num_layers - 1` hidden layers (Linear, ReLU, dropout) and a final 2-class layer with log-softmax. You can set `hidden_size`, `num_layers`, `dropout`, `frozen`, `random_init`, `embed_dim` and `bpe`. The embedding layer is chosen in this order:
    1. `bpe` given: random `nn.Embedding(len(bpe), embed_dim, padding_idx=0)` over the subword vocab. `word_embeddings` can be left out.
    2. `random_init=True` (the default): random `nn.Embedding` of size `embed_dim` (default 50) over the GloVe word vocab.
    3. Otherwise: the pretrained GloVe vectors, frozen unless `frozen=False`.
  - Averaging uses a mask `word_idxs != 0`: embeddings are multiplied by the mask, summed over the sequence, and divided by the number of real tokens (clamped to at least 1). This is why PAD must be index 0 in every vocabulary.
- **`bpe.py`**: Byte-pair encoding, following Sennrich et al. (2016).
  - `build_word_vocab` counts each lowercased word in `train.txt` and stores it as space-separated characters plus an end-of-word marker, e.g. `low` becomes `"l o w </w>"`. Lowercasing and splitting match `read_sentiment_examples`. The file is read directly, so `bpe.py` does not need torch.
  - `get_stats` counts every adjacent symbol pair, weighted by word frequency. `merge_vocab` replaces every standalone occurrence of the chosen pair with the joined symbol, using a regex with whitespace lookarounds so it never merges inside an existing symbol.
  - `learn_bpe(infile, vocab_size)` starts with every character seen in training (plus `</w>`) as the vocab, then repeatedly merges the most frequent pair. Each merge adds exactly one token. It stops when the vocab reaches `vocab_size`, or when no pairs are left. It returns the ordered list of merges and the sorted tokens.
  - `encode_word` splits a new word by starting from characters + `</w>` and replaying the merges in the order they were learned. `encode_sentence` does this for every word in a list.
  - `BPETokenizer(infile, vocab_size)` wraps the above: it stores the merges and a `tok2idx` map with `PAD` = 0 and `UNK` = 1, followed by the learned tokens (so `len(bpe)` is `vocab_size + 2`). `encode(words)` returns subword indices, maps unseen characters to `UNK`, and caches each word's split.
- **`sentiment_data.py`**
  - `read_sentiment_examples` reads lines of the form `label<TAB>sentence`, lowercases and tokenizes them, and returns `SentimentExample`s.
  - `read_word_embeddings` loads a GloVe text file into a `WordEmbeddings` object, with zero vectors for `PAD` and `UNK` at indices 0 and 1.
- **`utils.py`**: `Indexer`, a two-way mapping between words and integer ids.
- **`data/`**
  - `train.txt` and `dev.txt` hold the labeled sentences (1 = positive, 0 = negative).
  - `glove.6B.{50,300}d-relativized.txt` hold GloVe vectors, trimmed to the words that appear in this dataset.

## Changing DAN settings

These settings are in the `DAN` branch of `main.py` (the `make_model` lambdas):

- **GloVe vs. random embeddings:** `DAN` defaults to `random_init=True`, which trains new 50d embeddings over the GloVe word vocab and ignores the GloVe vectors. Pass `random_init=False` to use the pretrained vectors.
- **50d vs. 300d GloVe:** `main.py` loads `data/glove.6B.300d-relativized.txt`. Change the file name passed to `read_word_embeddings` to use the 50d file.
- **Fine-tune the GloVe vectors:** pass `random_init=False, frozen=False` to `DAN(...)`. Random embeddings are always trainable.
- **Embedding size:** pass `embed_dim=...` to `DAN(...)`. This applies to random and BPE embeddings.
- **Dropout:** pass `dropout=0.3` (or another rate) to `DAN(...)`.

## DAN-BPE details

- **Vocab size:** set with `--vocab_size`. The vocab counts the starting characters (65 in `train.txt`) plus one token per merge, so `--vocab_size 1000` means 935 merges. PAD and UNK are added on top.
- **Train only:** BPE is learned on `train.txt` and applied unchanged to `dev.txt`, so dev never leaks into the vocabulary. Dev words that were never seen in training still split into known subwords; only characters missing from training become `UNK`.
- **Embeddings:** always randomly initialized and trained, because GloVe has no vectors for subwords. Change their size with `embed_dim` in the `DAN-BPE` branch of `main.py`.
- **Sequence length:** sentences are longer in subwords than in words, especially for small vocab sizes, so each padded batch tensor is wider than in `DAN`.
- **Speed:** learning BPE re-scans every unique training word for each merge. A vocab size of 1000 takes about a minute; sizes like 5000 to 10000 take several minutes. This happens once, at the start of each run.
