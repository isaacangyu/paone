import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sentiment_data import read_sentiment_examples, WordEmbeddings
from bpe import BPETokenizer


# Dataset class that turns each sentence into a padded tensor of word indices
class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, word_embeddings: WordEmbeddings):
        self.examples = read_sentiment_examples(infile)
        indexer = word_embeddings.word_indexer
        pad_idx = indexer.index_of("PAD")
        unk_idx = indexer.index_of("UNK")

        # Map words to indices, using UNK for words without an embedding
        idx_seqs = []
        for ex in self.examples:
            idxs = [indexer.index_of(w) for w in ex.words]
            idx_seqs.append([i if i != -1 else unk_idx for i in idxs])

        # Pad every sentence to the longest one so they can be batched
        max_len = max(len(seq) for seq in idx_seqs)
        padded = [seq + [pad_idx] * (max_len - len(seq)) for seq in idx_seqs]

        self.word_idxs = torch.tensor(padded, dtype=torch.long)
        self.labels = torch.tensor([ex.label for ex in self.examples], dtype=torch.long)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.word_idxs[idx], self.labels[idx]


# Same as above, but sentences are split into BPE subwords instead of whole words
class SentimentDatasetDANBPE(Dataset):
    def __init__(self, infile, bpe: BPETokenizer):
        self.examples = read_sentiment_examples(infile)
        idx_seqs = [bpe.encode(ex.words) for ex in self.examples]

        max_len = max(len(seq) for seq in idx_seqs)
        padded = [seq + [0] * (max_len - len(seq)) for seq in idx_seqs]

        self.word_idxs = torch.tensor(padded, dtype=torch.long)
        self.labels = torch.tensor([ex.label for ex in self.examples], dtype=torch.long)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.word_idxs[idx], self.labels[idx]


# Deep Averaging Network: average the word embeddings, then feed through a feedforward network
class DAN(nn.Module):
    def __init__(self, word_embeddings: WordEmbeddings = None, hidden_size=100, num_layers=2, dropout=0.0, frozen=True,
                 random_init=True, embed_dim=50, bpe: BPETokenizer = None):
        super().__init__()
        if bpe is not None:
            # Randomly initialized embeddings over the BPE subword vocabulary (GloVe has no subword vectors)
            input_size = embed_dim
            self.embedding = nn.Embedding(len(bpe), input_size, padding_idx=0)
        elif random_init:
            # Randomly initialized embeddings over the same vocabulary, always trainable; PAD stays a zero vector
            vocab_size = len(word_embeddings.word_indexer)
            input_size = embed_dim or word_embeddings.get_embedding_length()
            self.embedding = nn.Embedding(vocab_size, input_size, padding_idx=0)
        else:
            self.embedding = word_embeddings.get_initialized_embedding_layer(frozen=frozen)
            input_size = word_embeddings.get_embedding_length()

        sizes = [input_size] + [hidden_size] * (num_layers - 1)
        self.hidden_layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i + 1]) for i in range(num_layers - 1)])
        self.output_layer = nn.Linear(sizes[-1], 2)
        self.dropout = nn.Dropout(dropout)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, word_idxs):
        # Average only over real tokens, ignoring PAD (index 0)
        mask = (word_idxs != 0).unsqueeze(-1).float()
        word_embeddings = self.embedding(word_idxs)
        x = (word_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        for layer in self.hidden_layers:
            x = self.dropout(F.relu(layer(x)))
        x = self.output_layer(x)
        return self.log_softmax(x)
